import tensorflow as tf

from graphs.adversarial.AAE_graph import inference_discriminate_encode_fn
from graphs.builder import layer_stuffing, clone_model
from training.autoencoding_basic.transformative.AE import autoencoder
from training.callbacks.early_stopping import EarlyStopping
from utils.swe.codes import copy_fn


class AAE(autoencoder):
    def __init__(
            self,
            strategy=None,
            **kwargs
    ):
        self.strategy = strategy
        self.input_scale = 1
        autoencoder.__init__(
            self,
            **kwargs
        )
        self.ONES = tf.ones(shape=[self.batch_size, 1])
        self.ZEROS = tf.zeros(shape=[self.batch_size, 1])

        self.adversarial_models = {
            'inference_discriminator_real':
                {
                    'variable': None,
                    'adversarial_item': 'generative',
                    'adversarial_value': self.ONES
                },
            'inference_discriminator_fake':
                {
                    'variable': None,
                    'adversarial_item': 'generative',
                    'adversarial_value': self.ZEROS
                },
            'inference_generator_fake':
                {
                    'variable': None,
                    'adversarial_item': 'generative',
                    'adversarial_value': self.ONES
                }
        }

    # combined models special
    def adversarial_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}


    def get_discriminators(self):
        return {k: model['variable'] for k, model in self.adversarial_models.items()}

    def create_batch_cast(self, models):
        def batch_cast_fn(xt0, xt1):
            if self.input_kw:
                xt0 = tf.cast(xt0[self.input_kw], dtype=tf.float32) / self.input_scale
                xt1 = tf.cast(xt1[self.input_kw], dtype=tf.float32) / self.input_scale
            else:
                xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
                xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale
            for k, model in models.items():
                if model['adversarial_item'] == 'inference':
                    outputs_dict = {k + '_outputs': model['adversarial_value']}
                    return {'inference_inputs': xt0}, outputs_dict
                else:
                    outputs_dict = {k + '_outputs': model['adversarial_value']}
                    encoded = autoencoder.__encode__(self, inputs={'inference_inputs': xt0})
                    return {'generative_inputs': encoded['z_latents']}, outputs_dict

        return batch_cast_fn

    def create_batch_cast_orig(self, models):
        def batch_cast_fn_orig(xt0, xt1):
            xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
            xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale
            encoded = autoencoder.__encode__(self, inputs={'inference_inputs': xt0})
            outputs_dict = {k + '_outputs': model['adversarial_value'] for k, model in models.items()}
            outputs_dict = {'x_logits': encoded['z_latents'], **outputs_dict}

            return {'inference_inputs': xt0, 'generative_inputs': encoded['z_latents']}, outputs_dict

        return batch_cast_fn_orig

    # override function
    def fit(
            self,
            x,
            validation_data=None,
            **kwargs
    ):
        print()
        print(f'training {autoencoder}')
        # 1- train the basic basicAE
        autoencoder.fit(
            self,
            x=x,
            validation_data=validation_data,
            **kwargs
        )


        def create_discriminator():
            for model in self.get_variables().values():
                layer_stuffing(model)

            for k, model in self.adversarial_models.items():
                model['variable'] = clone_model(old_model=self.get_variables()[model['adversarial_item']],  new_name=k,
                                                restore=self.filepath)

        # 2- create a latents discriminator
        if self.strategy:
            with self.strategy:
                create_discriminator()
        else:
            create_discriminator()

        # 3- clone autoencoder variables
        self.ae_get_variables = copy_fn(self.get_variables)

        # 4- switch to discriminate
        if self.strategy:
            if self.strategy:
                self.discriminators_compile()
        else:
            self.discriminators_compile()

        verbose = kwargs.pop('verbose')
        callbacks = kwargs.pop('callbacks')
        kwargs.pop('input_kw')

        for k, model in self.adversarial_models.items():
            print()
            print(f'training {k}')
            # 5- train the latents discriminator
            model['variable'].fit(
                x=x.map(self.create_batch_cast({k: model})),
                validation_data=None if validation_data is None else validation_data.map(self.create_batch_cast({k: model})),
                callbacks=[EarlyStopping()],
                verbose=1,
                **kwargs
            )

        kwargs['verbose'] = verbose
        kwargs['callbacks'] = callbacks

        # 6- connect all for inference_adversarial training
        if self.strategy:
            if self.strategy:
                self.__models_init__()
        else:
            self.__models_init__()

        print()
        print('training adversarial models')
        cbs = [cb for cb in callbacks or [] if isinstance(cb, tf.keras.callbacks.CSVLogger)]
        for cb in cbs:
            cb.filename = cb.filename.split('.csv')[0] + '_together.csv'
            mertic_names = [fn for sublist in [[k + '_' + fn.__name__ for fn in v] for k, v in self.ae_metrics.items()]
                            for fn in sublist]
            cb.keys = ['loss'] + [fn+'_loss' for fn in self._AA.output_names] + mertic_names
            cb.append_header = cb.keys

        # 7- training together
        self._AA.fit(
            x=x.map(self.create_batch_cast_orig(self.adversarial_models)),
            validation_data=None if validation_data is None else \
                validation_data.map(self.create_batch_cast_orig(self.adversarial_models)),
            **kwargs
        )

    def __models_init__(self):
        self.get_variables = self.adversarial_get_variables
        self.encode_fn = inference_discriminate_encode_fn
        inputs_dict= {
            'inference_inputs': self.get_variables()['inference'].inputs[0]
        }
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(encoded['z_latents'])

        outputs_dict = {k+'_predictions': encoded[k+'_predictions'] for k in self.adversarial_models.keys()}
        outputs_dict = {'x_logits': x_logits, **outputs_dict}

        self._AA = tf.keras.Model(
            name='adversarial_model',
            inputs= inputs_dict,
            outputs=outputs_dict
        )

        for i, outputs_dict in enumerate(self._AA.output_names):
            if 'x_logits' in outputs_dict:
                self._AA.output_names[i] = 'x_logits'
            for k in self.adversarial_models.keys():
                if k in outputs_dict:
                    self._AA.output_names[i] = k+'_outputs'

        generator_weight = self.adversarial_weights['generator_weight']
        discriminator_weight = self.adversarial_weights['discriminator_weight']
        generator_losses = [k for k in self.adversarial_losses.keys() if 'generator' in k]
        dlen = len(self.adversarial_losses)-len(generator_losses)
        aeloss_weights = {k: (1-discriminator_weight)*(1-generator_weight)/len(self.ae_losses) for k in self.ae_losses.keys()}
        gloss_weights = {k: (1-discriminator_weight)*(generator_weight)/len(generator_losses) for k in generator_losses}
        discriminator_weights = {k:  discriminator_weight/dlen for k in self.adversarial_losses.keys() if k not in generator_losses}
        adversarial_losses = {k: fn() for k, fn in self.adversarial_losses.items()}
        self._AA.compile(
            optimizer=self.optimizer,
            loss={**self.ae_losses, **adversarial_losses},
            metrics=self.ae_metrics,
            loss_weights={**aeloss_weights, **gloss_weights, **discriminator_weights}
        )

        self._AA.generate_sample = self.generate_sample
        self._AA.get_variable = self.get_variable
        self._AA.inputs_shape = self.get_inputs_shape()
        self._AA.get_inputs_shape = self.get_inputs_shape
        self._AA.latents_dim = self.latents_dim
        self._AA.save = self.save

        print(self._AA.summary())

    # override function
    def compile(
            self,
            adversarial_losses,
            adversarial_weights,
            **kwargs
    ):
        self.adversarial_losses=adversarial_losses
        self.adversarial_weights=adversarial_weights
        autoencoder.compile(
            self,
            **kwargs
        )

    def discriminators_compile(self, **kwargs):
        for k, model in self.adversarial_models.items():
            model['variable'].compile(
                optimizer=self.optimizer,
                loss=self.adversarial_losses[k+'_outputs']()
            )

            print(model['variable'].summary())
