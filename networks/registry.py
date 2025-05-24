"""Network registry module for registering and retrieving network architectures."""

from typing import Any, Callable, Dict, Type, Union
import torch.nn as nn
import yaml

from networks.EncoderDecoder.edclean import Generator as EDCleanGenerator
from networks.networks import (
    ResnetGenerator, UnetGenerator, UnetGeneratorA,
    NLayerDiscriminator, PixelDiscriminator
)
from networks.cyclegan.models import (
    GeneratorResNet as CycleGANGenerator,
    Discriminator as CycleGANDiscriminator
)


class NetworkRegistry:
    """Registry for network architectures."""
    
    def __init__(self):
        self._generators: Dict[str, Type[nn.Module]] = {}
        self._discriminators: Dict[str, Type[nn.Module]] = {}
    
    def register_generator(self, name: str, generator_cls: Type[nn.Module]) -> None:
        """Register a generator class."""
        self._generators[name] = generator_cls
    
    def register_discriminator(self, name: str, discriminator_cls: Type[nn.Module]) -> None:
        """Register a discriminator class."""
        self._discriminators[name] = discriminator_cls
    
    def get_generator(self, name: str, **kwargs: Any) -> nn.Module:
        """Get a generator instance by name."""
        if name.startswith('de'):  # DeScarGan
            return self._generators['edclean'](
                n_channels=kwargs.get('input_nc'),
                out_channels=kwargs.get('output_nc'),
                batch_norm=kwargs.get('batch_norm', {'batch': True, 'none': False}[kwargs.get('norm', 'batch')]),
                final=kwargs.get('final'),
                mc=kwargs.get('mc', False)
            )
        
        elif name.startswith('ed'):  # EncoderDecoder
            return self._generators['edclean'](
                n_channels=kwargs.get('input_nc'),
                out_channels=kwargs.get('output_nc'),
                nf=kwargs.get('ngf'),
                norm_type=kwargs.get('norm'),
                final=kwargs.get('final'),
                mc=kwargs.get('mc', False)
            )
        
        elif name.startswith('ldm'):  # LDM
            try:
                # Import dynamically since LDM is optional
                from networks.ldm.ae import AE
                with open(f'networks/ldm/{name}.yaml', "r") as f:
                    config = yaml.load(f, Loader=yaml.Loader)
                ddconfig = config['model']['params']["ddconfig"]
                return AE(ddconfig)
            except ImportError:
                raise ValueError('LDM module not found. Please install it first.')
        
        elif name.startswith('resnet'):
            n_blocks = int(name.split('_')[-1]) if '_' in name else 9
            return self._generators['resnet'](
                input_nc=kwargs.get('input_nc'),
                output_nc=kwargs.get('output_nc'),
                ngf=kwargs.get('ngf'),
                norm_layer=kwargs.get('norm_layer'),
                use_dropout=kwargs.get('use_dropout'),
                n_blocks=n_blocks,
                final=kwargs.get('final')
            )
        
        elif name.startswith('unet'):
            num_downs = int(name.split('_')[-1]) if '_' in name else 7
            return self._generators['unet'](
                input_nc=kwargs.get('input_nc'),
                output_nc=kwargs.get('output_nc'),
                num_downs=num_downs,
                ngf=kwargs.get('ngf'),
                norm_layer=kwargs.get('norm_layer'),
                use_dropout=kwargs.get('use_dropout'),
                final=kwargs.get('final')
            )
        
        elif name.startswith('uneta'):
            num_downs = int(name.split('_')[-1]) if '_' in name else 7
            return self._generators['uneta'](
                input_nc=kwargs.get('input_nc'),
                output_nc=kwargs.get('output_nc'),
                num_downs=num_downs,
                ngf=kwargs.get('ngf'),
                norm_layer=kwargs.get('norm_layer'),
                use_dropout=kwargs.get('use_dropout'),
                final=kwargs.get('final')
            )
        
        elif name.startswith('cyclegan'):
            return self._generators['cyclegan'](
                input_shape=(kwargs.get('input_nc'), 256, 256),
                num_residual_blocks=kwargs.get('n_residual_blocks', 9)
            )
            
        raise ValueError(f'Generator architecture {name} not recognized')
    
    def get_discriminator(self, name: str, **kwargs: Any) -> nn.Module:
        """Get a discriminator instance by name."""
        if name.startswith('patch'):
            patch_size = int(name.split('_')[-1])
            return self._discriminators['cyclegan'](
                input_shape=(kwargs.get('input_nc', 1), 256, 256),
                patch=patch_size,
                ndf=kwargs.get('ndf', 64)
            )
        
        elif name.startswith('nlayer'):
            n_layers = int(name.split('_')[-1]) if '_' in name else 3
            return self._discriminators['nlayer'](
                input_nc=kwargs.get('input_nc'),
                ndf=kwargs.get('ndf'),
                n_layers=n_layers,
                norm_layer=kwargs.get('norm_layer')
            )
        
        elif name == 'pixel':
            return self._discriminators['pixel'](
                input_nc=kwargs.get('input_nc'),
                ndf=kwargs.get('ndf'),
                norm_layer=kwargs.get('norm_layer')
            )
            
        raise ValueError(f'Discriminator architecture {name} not recognized')


# Create global registry instance
network_registry = NetworkRegistry()

# Register generator architectures
network_registry.register_generator('edclean', EDCleanGenerator)
network_registry.register_generator('resnet', ResnetGenerator)
network_registry.register_generator('unet', UnetGenerator)
network_registry.register_generator('uneta', UnetGeneratorA)
network_registry.register_generator('cyclegan', CycleGANGenerator)

# Register discriminator architectures
network_registry.register_discriminator('cyclegan', CycleGANDiscriminator)
network_registry.register_discriminator('nlayer', NLayerDiscriminator)
network_registry.register_discriminator('pixel', PixelDiscriminator)
