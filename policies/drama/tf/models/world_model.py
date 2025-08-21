"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
import numpy as np

import gymnasium as gym
import tree  # pip install dm_tree
from einops import rearrange, repeat, reduce

from policies.drama.tf.models.components.continue_predictor import (
    ContinuePredictor,
)
from policies.drama.tf.models.components.dynamics_predictor import (
    DynamicsPredictor,
)
from policies.drama.tf.models.components.mlp import MLP
from policies.drama.tf.models.components.representation_layer import (
    RepresentationLayer,
)
from policies.drama.tf.models.components.reward_predictor import (
    RewardPredictor,
)
from policies.drama.tf.models.components.sequence_model import (
    SequenceModel,
)
from policies.drama.utils import get_gru_units
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import symlog

from policies.drama.mamba_ssm import MambaWrapperModel, MambaConfig, InferenceParams, update_graph_cache

from policies.drama.utils import (
    get_dense_hidden_units,
    get_num_dense_layers,
    get_num_z_categoricals,
    get_num_z_categoricals,
    get_num_z_classes,
)

_, tf, _ = try_import_tf()
import tensorflow_probability as tfp

class DistHead(tf.keras.layers.Layer):
    def __init__(self, hidden_state_dim, categorical_dim, class_dim, unimix_ratio=0.01, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.stoch_dim = categorical_dim
        self.class_dim = class_dim
        self.unimix_ratio = unimix_ratio

        # Define linear layers
        self.post_head = tf.keras.layers.Dense(categorical_dim * class_dim, dtype=dtype)
        self.prior_head = tf.keras.layers.Dense(categorical_dim * class_dim, dtype=dtype)

    def unimix(self, logits, mixing_ratio=0.01):
        if mixing_ratio > 0:
            probs = tf.nn.softmax(logits, axis=-1)
            uniform = tf.ones_like(probs) / tf.cast(self.stoch_dim, self.dtype)
            mixed_probs = mixing_ratio * uniform + (1.0 - mixing_ratio) * probs
            logits = tf.math.log(mixed_probs + 1e-8)  # add small value to prevent log(0)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)  # shape: [B, L, K*C]
        logits = tf.reshape(logits, shape=(-1, tf.shape(logits)[1], self.stoch_dim, self.class_dim))  # [B, L, K, C]
        logits = self.unimix(logits, self.unimix_ratio)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)  # shape: [B, L, K*C]
        logits = tf.reshape(logits, shape=(-1, tf.shape(logits)[1], self.stoch_dim, self.class_dim))  # [B, L, K, C]
        logits = self.unimix(logits, self.unimix_ratio)
        return logits

@tf.function
def straight_through_gradient(logits, sample_mode="random_sample"):
    dist = tfp.distributions.OneHotCategorical(logits=logits)
    probs = dist.probs_parameter()

    if sample_mode == "random_sample":
        sample = dist.sample()
        # Straight-through gradient: pass forward `sample`, but use gradient of `probs`
        sample = tf.cast(dist.sample(), dtype=probs.dtype)
        sample = tf.stop_gradient(sample - probs) + probs

    elif sample_mode == "mode":
        # argmax gives the mode
        sample = tf.one_hot(tf.argmax(logits, axis=-1), depth=logits.shape[-1])

    elif sample_mode == "probs":
        sample = dist.probs

    return sample

def flatten_sample(sample):
        return rearrange(sample, "B L K C -> B L (K C)")

class WorldModel(tf.keras.Model):
    """WorldModel component of [1] w/ encoder, decoder, RSSM, reward/cont. predictors.

    See eq. 3 of [1] for all components and their respective in- and outputs.
    Note that in the paper, the "encoder" includes both the raw encoder plus the
    "posterior net", which produces posterior z-states from observations and h-states.

    Note: The "internal state" of the world model always consists of:
    The actions `a` (initially, this is a zeroed-out action), `h`-states (deterministic,
    continuous), and `z`-states (stochastic, discrete).
    There are two versions of z-states: "posterior" for world model training and "prior"
    for creating the dream data.

    Initial internal state values (`a`, `h`, and `z`) are inserted where ever a new
    episode starts within a batch row OR at the beginning of each train batch's B rows,
    regardless of whether there was an actual episode boundary or not. Thus, internal
    states are not required to be stored in or retrieved from the replay buffer AND
    retrieved batches from the buffer must not be zero padded.

    Initial `a` is the zero "one hot" action, e.g. [0.0, 0.0] for Discrete(2), initial
    `h` is a separate learned variable, and initial `z` are computed by the "dynamics"
    (or "prior") net, using only the initial-h state as input.
    """

    def __init__(
        self,
        *,
        model_size: str = "XS",
        observation_space: gym.Space,
        action_space: gym.Space,
        batch_length_T: int = 64,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        num_gru_units: Optional[int] = None,
        symlog_obs: bool = True,
    ):
        """Initializes a WorldModel instance.

        Args:
             model_size: The "Model Size" used according to [1] Appendinx B.
                Use None for manually setting the different network sizes.
             observation_space: The observation space of the environment used.
             action_space: The action space of the environment used.
             batch_length_T: The length (T) of the sequences used for training. The
                actual shape of the input data (e.g. rewards) is then: [B, T, ...],
                where B is the "batch size", T is the "batch length" (this arg) and
                "..." is the dimension of the data (e.g. (64, 64, 3) for Atari image
                observations). Note that a single row (within a batch) may contain data
                from different episodes, but an already on-going episode is always
                finished, before a new one starts within the same row.
            encoder: The encoder Model taking observations as inputs and
                outputting a 1D latent vector that will be used as input into the
                posterior net (z-posterior state generating layer). Inputs are symlogged
                if inputs are NOT images. For images, we use normalization between -1.0
                and 1.0 (x / 128 - 1.0)
            decoder: The decoder Model taking h- and z-states as inputs and generating
                a (possibly symlogged) predicted observation. Note that for images,
                the last decoder layer produces the exact, normalized pixel values
                (not a Gaussian as described in [1]!).
            num_gru_units: The number of GRU units to use. If None, use
                `model_size` to figure out this parameter.
            symlog_obs: Whether to predict decoded observations in symlog space.
                This should be False for image based observations.
                According to the paper [1] Appendix E: "NoObsSymlog: This ablation
                removes the symlog encoding of inputs to the world model and also
                changes the symlog MSE loss in the decoder to a simple MSE loss.
                *Because symlog encoding is only used for vector observations*, this
                ablation is equivalent to Drama on purely image-based environments".
        """
        super().__init__(name="world_model")

        self.model_size = model_size
        self.batch_length_T = batch_length_T
        self.symlog_obs = symlog_obs
        self.observation_space = observation_space
        self.action_space = action_space
        self._comp_dtype = (
            tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        )

        # Encoder (latent 1D vector generator) (xt -> lt).
        self.encoder = encoder

        self.dist_head = DistHead(
            hidden_state_dim=get_dense_hidden_units(self.model_size), # EMRAN this isn't exactly right, might need unique function size
            categorical_dim=get_num_z_categoricals(self.model_size), # EMRAN copied what was used in Representation layer, size may not be correct 
            class_dim=get_num_z_classes(self.model_size), # EMRAN copied what was used in Representation layer, size may not be correct
            dtype=self.dtype # EMRAN idk if this is right bruh
        )

        # # Posterior predictor consisting of an MLP and a RepresentationLayer:
        # # [ht, lt] -> zt.
        # self.posterior_mlp = MLP(
        #     model_size=self.model_size,
        #     output_layer_size=None,
        #     # In Danijar's code, the posterior predictor only has a single layer,
        #     # no matter the model size:
        #     num_dense_layers=1,
        #     name="posterior_mlp",
        # )
        # The (posterior) z-state generating layer.
        self.posterior_representation_layer = RepresentationLayer(
            model_size=self.model_size,
        )

        # Dynamics (prior z-state) predictor: ht -> z^t
        self.dynamics_predictor = DynamicsPredictor(model_size=self.model_size)

        # GRU for the RSSM: [at, ht, zt] -> ht+1
        self.num_gru_units = get_gru_units(
            model_size=self.model_size,
            override=num_gru_units,
        )
        # Initial h-state variable (learnt).
        # -> tanh(self.initial_h) -> deterministic state
        # Use our Dynamics predictor for initial stochastic state, BUT with greedy
        # (mode) instead of sampling.
        self.initial_h = tf.Variable(
            tf.zeros(shape=(self.num_gru_units,)),
            trainable=True,
            name="initial_h",
        )
        
        # The actual sequence model containing the GRU layer.
        # self.sequence_model = SequenceModel(
        #     model_size=self.model_size,
        #     action_space=self.action_space,
        #     num_gru_units=self.num_gru_units,
        # )
        mamba_config = MambaConfig(
            model_size=self.model_size,
            action_dim=self.action_space.n
                            if isinstance(self.action_space, gym.spaces.Discrete)
                            else int(np.prod(self.action_space.shape)),
            ssm_cfg={
                'd_state': 16, # config.Models.WorldModel.Mamba.ssm_cfg.d_state, 
                'layer': 'Mamba2'}
            )
        self.sequence_model = MambaWrapperModel(mamba_config)
        
        # Reward Predictor: [ht, zt] -> rt.
        self.reward_predictor = RewardPredictor(model_size=self.model_size)
        # Continue Predictor: [ht, zt] -> ct.
        self.continue_predictor = ContinuePredictor(model_size=self.model_size)

        # Decoder: [ht, zt] -> x^t.
        self.decoder = decoder

        # Trace self.call.
        self.forward_train = tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, None] + list(self.observation_space.shape)),
                tf.TensorSpec(
                    shape=[None, None]
                    + (
                        [self.action_space.n]
                        if isinstance(action_space, gym.spaces.Discrete)
                        else list(self.action_space.shape)
                    )
                ),
                tf.TensorSpec(shape=[None, None], dtype=tf.bool),
            ]
        )(self.forward_train)

    @tf.function
    def get_initial_state(self):
        """Returns the (current) initial state of the world model (h- and z-states).

        An initial state is generated using the tanh of the (learned) h-state variable
        and the dynamics predictor (or "prior net") to compute z^0 from h0. In this last
        step, it is important that we do NOT sample the z^-state (as we would usually
        do during dreaming), but rather take the mode (argmax, then one-hot again).
        """
        h = tf.expand_dims(tf.math.tanh(tf.cast(self.initial_h, self._comp_dtype)), 0)
        # Use the mode, NOT a sample for the initial z-state.
        _, z_probs = self.dynamics_predictor(h)
        z = tf.argmax(z_probs, axis=-1)
        z = tf.one_hot(z, depth=z_probs.shape[-1], dtype=self._comp_dtype)

        return {"h": h, "z": z}

    def forward_inference(self, observations, previous_states, is_first, training=None):
        """Performs a forward step for inference (e.g. environment stepping).

        Works analogous to `forward_train`, except that all inputs are provided
        for a single timestep in the shape of [B, ...] (no time dimension!).

        Args:
            observations: The batch (B, ...) of observations to be passed through
                the encoder network to yield the inputs to the representation layer
                (which then can compute the z-states).
            previous_states: A dict with `h`, `z`, and `a` keys mapping to the
                respective previous states/actions. All of the shape (B, ...), no time
                rank.
            is_first: The batch (B) of `is_first` flags.

        Returns:
            The next deterministic h-state (h(t+1)) as predicted by the sequence model.
        """
        # observations = tf.cast(observations, self._comp_dtype)

        # initial_states = tree.map_structure(
        #     lambda s: tf.repeat(s, tf.shape(observations)[0], axis=0),
        #     self.get_initial_state(),
        # )

        # # If first, mask it with initial state/actions.
        # previous_h = self._mask(previous_states["h"], 1.0 - is_first)  # zero out
        # previous_h = previous_h + self._mask(initial_states["h"], is_first)  # add init

        # previous_z = self._mask(previous_states["z"], 1.0 - is_first)  # zero out
        # previous_z = previous_z + self._mask(initial_states["z"], is_first)  # add init

        # # Zero out actions (no special learnt initial state).
        # previous_a = self._mask(previous_states["a"], 1.0 - is_first)

        # # Compute new states.
        # h = self.sequence_model(a=previous_a, h=previous_h, z=previous_z)
        # z = self.compute_posterior_z(observations=observations, initial_h=h)

        print("\n\n\n\n\n\n\nprevious_states:", previous_states)

        B, T, obs_dim = previous_states['context_obs'].shape
        B, T, action_dim = previous_states['context_action'].shape

        print("AFTER) observation:", tf.reshape(previous_states['context_obs'], [-1, obs_dim]))
        embedding = self.encoder(
            tf.cast(tf.reshape(previous_states['context_obs'], [-1, obs_dim]), self._comp_dtype)
        )
        embedding = tf.reshape(
            embedding, shape=tf.concat([[B, T], tf.shape(embedding)[1:]], axis=0)
        )
        print("AFTER) embedding:", embedding)
        post_logits = self.dist_head.forward_post(embedding)
        post_sample = straight_through_gradient(post_logits, sample_mode="random_sample")
        flattened_post = flatten_sample(post_sample)

        print("after embedding")

        dist_feat = self.sequence_model(flattened_post, previous_states['context_action'])
        print("after sequence model")
        last_dist_feat = dist_feat[:, -1:]
        print("last_dist_feat:", last_dist_feat)
        prior_logits = self.dist_head.forward_prior(last_dist_feat)
        prior_sample = straight_through_gradient(prior_logits, sample_mode="random_sample")
        flattened_prior = flatten_sample(prior_sample)
        

        print("returning dist_feat and prior logits")

        return dist_feat, flattened_prior

    def forward_train(self, observations, actions, is_first):
        """Performs a forward step for training.

        1) Forwards all observations [B, T, ...] through the encoder network to yield
        o_processed[B, T, ...].
        2) Uses initial state (h0/z^0/a0[B, 0, ...]) and sequence model (RSSM) to
        compute the first internal state (h1 and z^1).
        3) Uses action a[B, 1, ...], z[B, 1, ...] and h[B, 1, ...] to compute the
        next h-state (h[B, 2, ...]), etc..
        4) Repeats 2) and 3) until t=T.
        5) Uses all h[B, T, ...] and z[B, T, ...] to compute predicted/reconstructed
        observations, rewards, and continue signals.
        6) Returns predictions from 5) along with all z-states z[B, T, ...] and
        the final h-state (h[B, ...] for t=T).

        Should we encounter is_first=True flags in the middle of a batch row (somewhere
        within an ongoing sequence of length T), we insert this world model's initial
        state again (zero-action, learned init h-state, and prior-computed z^) and
        simply continue (no zero-padding).

        Args:
            observations: The batch (B, T, ...) of observations to be passed through
                the encoder network to yield the inputs to the representation layer
                (which then can compute the posterior z-states).
            actions: The batch (B, T, ...) of actions to be used in combination with
                h-states and computed z-states to yield the next h-states.
            is_first: The batch (B, T) of `is_first` flags.
        """

        if self.symlog_obs:
            observations = symlog(observations)

        # Compute bare encoder outs (not z; this is done later with involvement of the
        # sequence model and the h-states).
        # Fold time dimension for CNN pass.
        shape = tf.shape(observations)
        B, T = shape[0], shape[1]
        observations = tf.reshape(
            observations, shape=tf.concat([[-1], shape[2:]], axis=0)
        )
        print("BEFORE) observation:", observations)
        embedding = self.encoder(tf.cast(observations, self._comp_dtype))
        embedding = tf.reshape(
            embedding, shape=tf.concat([[B, T], tf.shape(embedding)[1:]], axis=0)
        )
        print("BEFORE) embedding:", embedding)
        post_logits = self.dist_head.forward_post(embedding)
        post_sample = straight_through_gradient(post_logits, sample_mode="random_sample")
        flattened_post = flatten_sample(post_sample)
        
        # decoding image
        obs_hat = self.decoder(flattened_post)
        obs_hat_BxT = tf.reshape(
            obs_hat, shape=tf.concat([[-1], shape[2:]], axis=0)
        )

        # dynamics model
        dist_feat = self.sequence_model(flattened_post, actions)
        prior_logits = self.dist_head.forward_prior(dist_feat)
        prior_sample = straight_through_gradient(prior_logits, sample_mode="random_sample")
        flattened_prior = flatten_sample(prior_sample)

        dist_feat_BxT = tf.reshape(
            dist_feat, shape=tf.concat([[-1], tf.shape(dist_feat)[2:]], axis=0)
        )

        # Compute (predicted) reward distributions.
        rewards, reward_logits = self.reward_predictor(dist_feat_BxT)

        # Compute (predicted) continue distributions.
        continues, continue_distribution = self.continue_predictor(dist_feat_BxT)
 
        # print("observations, obs_hat_BxT:", observations, obs_hat_BxT)
        # print("continues, continue_distribution:", continues, continue_distribution)
        # exit()

        # Return outputs for loss computation.
        # Note that all shapes are [BxT, ...] (time axis already folded).
        return {
            # Obs.
            "sampled_obs_symlog_BxT": observations,
            "obs_distribution_means_BxT": obs_hat_BxT, # EMRAN may need to cast tf.cast(obs_hat, tf.float32),
            # Rewards.
            "reward_logits_BxT": reward_logits,
            "rewards_BxT": rewards,
            # Continues.
            "continue_distribution_BxT": continue_distribution,
            "continues_BxT": continues,
            # Hidden states 
            "dist_feat": dist_feat,
            "flattened_prior": flattened_prior,
            "flattened_post": flattened_post,

            # EMRAN below is DreamerV3 stuff not used anymore
            # Deterministic, continuous h-states (t1 to T).
            # "h_states_BxT": None, # EMRAN used to be h_BxT 
            # # Sampled, discrete posterior z-states and their probs (t1 to T).
            # "z_posterior_states_BxT": None, # EMRAN used to be z_BxT
            # "z_posterior_probs_BxT": None, # EMRAN used to be z_posterior_probs
            # # Probs of the prior z-states (t1 to T).
            # "z_prior_probs_BxT": None, # EMRAN used to be z_prior_probs
        }

    def compute_posterior_z(self, observations, initial_h):
        # Compute bare encoder outputs (not including z, which is computed in next step
        # with involvement of the previous output (initial_h) of the sequence model).
        # encoder_outs=[B, ...]
        if self.symlog_obs:
            observations = symlog(observations)
        encoder_out = self.encoder(observations)
        # Concat encoder outs with the h-states.
        posterior_mlp_input = tf.concat([encoder_out, initial_h], axis=-1)
        # Compute z.
        repr_input = self.posterior_mlp(posterior_mlp_input)
        # Draw a z-sample.
        z_t, _ = self.posterior_representation_layer(repr_input)
        return z_t

    @staticmethod
    def _mask(value, mask):
        return tf.einsum("b...,b->b...", value, tf.cast(mask, value.dtype))
