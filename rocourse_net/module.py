# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_training_module.ipynb.

# %% ../nbs/00_training_module.ipynb 1
from __future__ import annotations
from relax.import_essentials import *
from relax.methods.counternet import (
    CounterNetTrainingModule, 
    CounterNetTrainingModuleConfigs, 
    partition_trainable_params,
    CounterNet,
    CounterNetConfigs
)
from relax.data import load_data
from relax.utils import grad_update
from copy import deepcopy
from functools import partial
import chex

# %% auto 0
__all__ = ['l_inf_proj', 'Attacker', 'RandomAttacker', 'BilevelAttacker', 'RoCourseNetTrainingConfigs',
           'RoCourseNetTrainingModule', 'RoCourseNetConfigs', 'RoCourseNet']

# %% ../nbs/00_training_module.ipynb 2
def l_inf_proj(x: jnp.ndarray, eps: float, cat_idx: Optional[int] = None):
    if cat_idx is None:
        return x.clip(-eps, eps)
    else:
        return jnp.concatenate([
            x[:, :cat_idx].clip(-eps, eps), # clip continuous features only
            x[:, cat_idx:]
        ], axis=-1)

# %% ../nbs/00_training_module.ipynb 3
def filter_params(params):
    return hk.data_structures.filter(
        lambda m, n, v: m == 'counter_net_model/Predictor/dense_block/linear' and n == 'w', params
    )

# %% ../nbs/00_training_module.ipynb 4
class Attacker(ABC):
    def __init__(
        self,
        keys: hk.PRNGSequence, # ignored
        pred_loss_fn: Callable[[hk.Params, random.PRNGKey, Tuple[jnp.DeviceArray, jnp.DeviceArray], bool], jnp.DeviceArray],
        adv_loss_fn, # ignored 
        n_steps: int, # attacker steps
        k: int, # inner steps
        epsilon: float,
        adv_lr: float,
        apply_fn: Callable, # apply_fn(x, cf, hard=False)
        cat_idx, # ignored
        check_assertions: bool = False,
    ):
        self.keys = keys
        self.pred_loss_fn = pred_loss_fn
        self.adv_loss_fn = adv_loss_fn
        self.n_steps = n_steps
        self.k = k
        self.epsilon = epsilon
        self.adv_lr = adv_lr
        self.apply_fn = apply_fn
        self.cat_idx = cat_idx
        self.check_assertions = check_assertions
    
    def step(
        self,
        params: hk.Params,
        x: jnp.ndarray, 
        y: jnp.ndarray,
    ) -> hk.Params:
        raise NotImplementedError
    
    __call__ = step

# %% ../nbs/00_training_module.ipynb 6
class RandomAttacker(Attacker):
    def step(self, params, rng_key, x, y):
        # init delta randomly
        delta = random.uniform(
            key=rng_key, shape=x.shape, 
            minval=-self.epsilon, maxval=self.epsilon)
        
        # create optimizer
        opt = optax.adam(learning_rate=self.adv_lr)
        opt_state = opt.init(params)

        for _ in range(self.n_steps):
            rng_key, *subkeys = random.split(rng_key, self.k + 2)
            x = self.apply_fn(x, x + delta, hard=False)

            for i in range(self.k):
                loss, grads = jax.value_and_grad(self.pred_loss_fn)(
                        params, subkeys[i], (x, y), False)
                params, opt_state = grad_update(grads, params, opt_state, opt)

            delta = random.uniform(
                key=subkeys[-1], shape=x.shape, 
                minval=-self.epsilon, maxval=self.epsilon)
            rng_key = subkeys[-1]
            
        return params

# %% ../nbs/00_training_module.ipynb 7
class BilevelAttacker(Attacker):
    def step(self, params: hk.Params, rng_key: random.PRNGKey, x: jnp.ndarray, y: jnp.ndarray) -> hk.Params:
        # alpha is the delta's learning rate
        alpha = self.epsilon * 2.5 / self.n_steps
        # init delta randomly
        delta = random.uniform(
            key=rng_key, shape=x.shape, 
            minval=-self.epsilon, maxval=self.epsilon)
        # create optimizer
        opt = optax.chain(
            optax.clip(1.0),
            optax.adam(learning_rate=self.adv_lr)
        )
        
        opt_state = opt.init(params)

        @partial(jax.jit, static_argnames=['opt'])
        def attacker_fn(
            delta: jnp.ndarray,
            params: hk.Params,
            opt_state: optax.OptState,
            rng_key: random.PRNGKey,
            batch: Tuple[jnp.ndarray, jnp.ndarray],
            opt: optax.GradientTransformation
        ):
            # def inner_step(states, k):
            #     params, opt_state, rng_key = states
            #     rng_key, sub_key = random.split(rng_key)
            #     _x = self.apply_fn(x, x + delta, hard=False)
            #     grads = jax.grad(self.pred_loss_fn)(params, rng_key, (_x, y), is_training=True)
            #     params, opt_state = grad_update(grads, params, opt_state, opt)
            #     return (params, opt_state, sub_key), None

            # x, y = batch
            # states = (params, opt_state, rng_key)
            # (params, opt_state, rng_key), _ = jax.lax.scan(inner_step, states, jnp.arange(self.k))
            # loss = self.adv_loss_fn(params, rng_key, x, is_training=False)
            # return loss, (params, opt_state)

            x, y = batch
            for _ in range(self.k):
                # inner unrolling steps
                _x = self.apply_fn(x, x + delta, hard=False)
                grads = jax.grad(self.pred_loss_fn)(params, rng_key, (_x, y), is_training=False)
                params, opt_state = grad_update(grads, params, opt_state, opt)

            loss = self.adv_loss_fn(params, rng_key, x, is_training=False)
            return loss, (params, opt_state)

        def attacker_step(states, k):
            delta, params, opt_state, rng_key = states
            rng_key, sub_key = random.split(rng_key)
            g, (params, opt_state) = jax.grad(attacker_fn, has_aux=True)(
                delta, params, opt_state, rng_key, (x, y), opt)

            g = jnp.clip(g, -1.0, 1.0)
            delta = delta + alpha * jnp.sign(g)
            delta = l_inf_proj(delta, self.epsilon, self.cat_idx)
            return (delta, params, opt_state, sub_key), None

        states = (delta, params, opt_state, rng_key)
        (delta, params, opt_state, rng_key), _ = jax.lax.scan(
            f=attacker_step, init=states, xs=jnp.arange(self.n_steps))

        # for i in range(self.n_steps):
        #     _, rng_key = random.split(rng_key)
            
            # g, (params, opt_state) = jax.grad(attacker_fn, has_aux=True)(
            #     delta, params, opt_state, rng_key, (x, y), opt)
            # delta = delta + alpha * jnp.sign(g)
            # delta = l_inf_proj(delta, self.epsilon, self.cat_idx)
        return params
        

# %% ../nbs/00_training_module.ipynb 9
class RoCourseNetTrainingConfigs(CounterNetTrainingModuleConfigs):
    epsilon: float = 0.1
    n_steps: int = 7
    k: int = 2
    adv_lr: float
    random_perturbation: bool = False
    seed: int = 42

    @property
    def keys(self):
        return hk.PRNGSequence(self.seed)

# %% ../nbs/00_training_module.ipynb 10
class RoCourseNetTrainingModule(CounterNetTrainingModule):
    name = "RoCourseNet"

    def __init__(self, m_configs: Dict[str, Any]):
        super().__init__(m_configs)
        self.configs = RoCourseNetTrainingConfigs(**m_configs)

    def init_net_opt(self, data_module, key):
        res = super().init_net_opt(data_module, key)
        if self.configs.random_perturbation:
            AdvCls = RandomAttacker
        else:
            AdvCls = BilevelAttacker
        
        self.attacker = AdvCls(
            keys=hk.PRNGSequence(self.configs.seed), 
            pred_loss_fn=self.pred_loss_fn,
            adv_loss_fn=self.adv_loss_fn,
            n_steps=self.configs.n_steps,
            k=self.configs.k,
            epsilon=self.configs.epsilon,
            adv_lr=self.configs.adv_lr,
            apply_fn=self._data_module.apply_constraints,
            cat_idx=self._data_module.cat_idx, 
        )

        return res
    
    def adv_loss_fn(
        self,
        params: hk.Params, 
        rng_key: random.PRNGKey, 
        x: jnp.DeviceArray, 
        is_training: bool = True
    ):
        y_pred, cf, cf_y = self.forward(params, rng_key, x, is_training)
        y_prime = 1 - jnp.round(y_pred)
        return self.loss_fn_2(cf_y, y_prime)

    def bilevel_adv_step(
        self,
        params: hk.Params,
        rng_key: random.PRNGKey,
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> hk.Params:
        # if self.configs.random_perturbation:
        #     AdvCls = RandomAttacker
        # else:
        #     AdvCls = BilevelAttacker

        return self.attacker.step(params, rng_key, *batch)

    def exp_loss_fn(
        self,
        trainable_params: hk.Params,
        non_trainable_params: hk.Params,
        aux_params: hk.Params,
        rng_key: random.PRNGKey,
        batch: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        is_training: bool = True
    ):
        # merge trainable and non_trainable params
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        x, y = batch
        y_pred, cf = self.net.apply(params, rng_key, x, is_training=is_training)
        cf = self._data_module.apply_constraints(x, cf, hard=not is_training)
        # compute cf_y on shifted model
        cf_y, _ = self.net.apply(aux_params, rng_key, cf, is_training=is_training)
        y_prime = 1 - jnp.round(y_pred)
        loss_2, loss_3 = self.loss_fn_2(cf_y, y_prime), self.loss_fn_3(x, cf)
        return self.configs.lambda_2 * loss_2 + self.configs.lambda_3 * loss_3

    def explainer_step(self, params, aux_params, opt_state, rng_key, batch):
        trainable_params, non_trainable_params = partition_trainable_params(
            params, trainable_name='counter_net_model/Explainer'
        )
        grads = jax.grad(self.exp_loss_fn)(
            trainable_params, non_trainable_params, aux_params, rng_key, batch)
        upt_trainable_params, opt_state = grad_update(
            grads, trainable_params, opt_state, self.opt_2)
        upt_params = hk.data_structures.merge(upt_trainable_params, non_trainable_params)
        return upt_params, opt_state

    @partial(jax.jit, static_argnames=['self'])
    def _training_step(self,
        params: hk.Params,
        opts_state: Tuple[optax.GradientTransformation, optax.GradientTransformation],
        rng_key: random.PRNGKey,
        batch: Tuple[jnp.array, jnp.array]
    ):
        opt_1_state, opt_2_state = opts_state
        params, opt_1_state = self._predictor_step(params, opt_1_state, rng_key, batch)
        aux_params = self.bilevel_adv_step(params, rng_key, batch)
        upt_params, opt_2_state = self.explainer_step(params, aux_params, opt_2_state, rng_key, batch)
        return upt_params, (opt_1_state, opt_2_state)

    def _training_step_logs(self, params, rng_key, batch):
        x, y = batch
        logs = super()._training_step_logs(params, rng_key, batch)
        adv_loss = self.adv_loss_fn(params, rng_key, x, is_training=False)
        logs.update({
            'train/adv_loss': adv_loss
        })
        return logs

# %% ../nbs/00_training_module.ipynb 15
class RoCourseNetConfigs(CounterNetConfigs):
    epsilon: float = 0.1
    n_steps: int = 7
    k: int = 2
    adv_lr: float = 0.03
    random_perturbation: bool = False
    seed: int = 42

    @property
    def keys(self):
        return hk.PRNGSequence(self.seed)

# %% ../nbs/00_training_module.ipynb 16
class RoCourseNet(CounterNet):
    name: str = "RoCourseNet"
    
    def __init__(self, m_configs: dict | RoCourseNetConfigs = None):
        super().__init__(m_configs)
        if m_configs is None:
            m_configs = RoCourseNetConfigs()
        self.module = RoCourseNetTrainingModule(m_configs)
