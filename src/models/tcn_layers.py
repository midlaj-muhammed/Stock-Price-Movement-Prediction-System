"""
Temporal Convolutional Network (TCN) layers implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.regularizers import l2
from typing import Optional, List

class TemporalBlock(Layer):
    """Temporal block for TCN architecture."""
    
    def __init__(
        self,
        n_outputs: int,
        kernel_size: int,
        strides: int,
        dilation_rate: int,
        dropout_rate: float = 0.0,
        trainable: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation: str = 'relu',
        kernel_initializer: str = 'he_normal',
        use_skip_connections: bool = True,
        l2_reg: float = 0.0,
        **kwargs
    ):
        """
        Initialize temporal block.
        
        Args:
            n_outputs: Number of output filters
            kernel_size: Kernel size for convolution
            strides: Stride for convolution
            dilation_rate: Dilation rate for convolution
            dropout_rate: Dropout rate
            trainable: Whether layer is trainable
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            activation: Activation function
            kernel_initializer: Kernel initializer
            use_skip_connections: Whether to use skip connections
            l2_reg: L2 regularization factor
        """
        super(TemporalBlock, self).__init__(**kwargs)
        
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_skip_connections = use_skip_connections
        self.l2_reg = l2_reg
        
        # Padding for causal convolution
        self.padding = (kernel_size - 1) * dilation_rate
        
        # First convolution
        self.conv1 = Conv1D(
            filters=n_outputs,
            kernel_size=kernel_size,
            strides=strides,
            padding='causal',
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            trainable=trainable
        )
        
        # Normalization layers
        if use_batch_norm:
            self.norm1 = BatchNormalization(trainable=trainable)
        elif use_layer_norm:
            self.norm1 = tf.keras.layers.LayerNormalization(trainable=trainable)
        else:
            self.norm1 = None
        
        # Activation
        self.activation1 = Activation(activation)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Second convolution
        self.conv2 = Conv1D(
            filters=n_outputs,
            kernel_size=kernel_size,
            strides=strides,
            padding='causal',
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            trainable=trainable
        )
        
        # Second normalization
        if use_batch_norm:
            self.norm2 = BatchNormalization(trainable=trainable)
        elif use_layer_norm:
            self.norm2 = tf.keras.layers.LayerNormalization(trainable=trainable)
        else:
            self.norm2 = None
        
        # Second activation
        self.activation2 = Activation(activation)
        
        # Second dropout
        self.dropout2 = Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Skip connection
        self.use_skip_connections = use_skip_connections
        if use_skip_connections:
            self.downsample = None
            self.add = Add()
            self.final_activation = Activation(activation)
    
    def build(self, input_shape):
        """Build the layer."""
        super(TemporalBlock, self).build(input_shape)
        
        # Check if we need a downsample layer for skip connection
        if self.use_skip_connections and input_shape[-1] != self.n_outputs:
            self.downsample = Conv1D(
                filters=self.n_outputs,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=l2(self.l2_reg) if self.l2_reg > 0 else None
            )
    
    def call(self, inputs, training=None):
        """Forward pass."""
        x = inputs
        
        # First convolution block
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x, training=training)
        x = self.activation1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x, training=training)
        
        # Second convolution block
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x, training=training)
        x = self.activation2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x, training=training)
        
        # Skip connection
        if self.use_skip_connections:
            residual = inputs
            if self.downsample is not None:
                residual = self.downsample(residual)
            
            x = self.add([x, residual])
            x = self.final_activation(x)
        
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super(TemporalBlock, self).get_config()
        config.update({
            'n_outputs': self.n_outputs,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'use_skip_connections': self.use_skip_connections,
            'l2_reg': self.l2_reg
        })
        return config

class TemporalConvNet(Layer):
    """Temporal Convolutional Network layer."""
    
    def __init__(
        self,
        num_channels: list,
        kernel_size: int = 2,
        dropout_rate: float = 0.0,
        trainable: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation: str = 'relu',
        use_skip_connections: bool = True,
        l2_reg: float = 0.0,
        **kwargs
    ):
        """
        Initialize TCN layer.
        
        Args:
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout_rate: Dropout rate
            trainable: Whether layer is trainable
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            activation: Activation function
            use_skip_connections: Whether to use skip connections
            l2_reg: L2 regularization factor
        """
        super(TemporalConvNet, self).__init__(**kwargs)
        
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.l2_reg = l2_reg
        
        # Create temporal blocks
        self.temporal_blocks = []
        for i, num_filters in enumerate(num_channels):
            dilation_rate = 2 ** i
            block = TemporalBlock(
                n_outputs=num_filters,
                kernel_size=kernel_size,
                strides=1,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate,
                trainable=trainable,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation=activation,
                use_skip_connections=use_skip_connections,
                l2_reg=l2_reg,
                name=f'temporal_block_{i}'
            )
            self.temporal_blocks.append(block)
    
    def call(self, inputs, training=None):
        """Forward pass."""
        x = inputs
        for block in self.temporal_blocks:
            x = block(x, training=training)
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super(TemporalConvNet, self).get_config()
        config.update({
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'activation': self.activation,
            'use_skip_connections': self.use_skip_connections,
            'l2_reg': self.l2_reg
        })
        return config
