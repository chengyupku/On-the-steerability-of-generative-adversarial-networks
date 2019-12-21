import tensorflow as tf
import tensorlayer as tensorlayer
from tensorlayer.layers import Input,Dense,DeConv2d,Reshape,BatchNorm2d, Conv2d,Flatten,MaxPool2d
def get_generator(shape,gf_dim=64):
    image_size=256
    s16=image_size//16
    w_init=tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1.,0.02)
    ni=Input(shape)
    nn=Dense(n_units=(gf_dim*16*s16*s16),W_init=w_init,b_init=None)(ni)
    nn=Reshape(shape=[-1,s16,s16,gf_dim*16])(nn)
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=DeConv2d(gf_dim*16,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+nn
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=DeConv2d(gf_dim*8,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+nn
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=DeConv2d(gf_dim*8,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+nn
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=DeConv2d(gf_dim*4,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+nn
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=DeConv2d(gf_dim*2,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+nn
    #non local block maybe bugs here
    f=Conv2d(gf_dim//4,(1,1),(1,1),W_init=w_init,b_init=None)(nn)
    f=MaxPool2d((2,2),(2,2))(f)
    g=Conv2d(gf_dim//4,(1,1),(1,1),W_init=w_init,b_init=None)(nn)
    h=Conv2d(gf_dim,(1,1),(1,1),W_init=w_init,b_init=None)(nn)
    h=MaxPool2d((2,2),(2,2))(h)
    s = tf.matmul(Reshape(shape=[g.shape[0],-1,g.shape[1]])(g), Reshape(shape=[f.shape[0],-1,f.shape[1]])(f), transpose_b=True)
    beta=tf.nn.softmax(s)
    o=tf.matmul(beta,Reshape(shape=[h.shape[0],-1,h.shape[1]])(h))
    o=Reshape(shape=[nn.shape[0],nn.shape[1],nn.shape[2],gf_dim])(o)
    o=Conv2d(gf_dim*2,(1,1),(1,1),W_init=w_init,b_init=None)(o)
    nn=nn+gamma_init*o
    #
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=DeConv2d(gf_dim,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+nn
    nn=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    nn=Conv2d(3, (3, 3), (1, 1), act=tf.nn.tanh, W_init=w_init)(nn)
    return tl.models.Model(inputs=ni,outputs=nn,name='generator')
def get_discriminator(shape,df_dim=64):
    w_init=tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1.,0.02)
    ni=Input(shape)
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(ni)
    n1=Conv2d(df_dim,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=n1+ni
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=Conv2d(df_dim*2,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=nn+n1;
    f=Conv2d(df_dim//4,(1,1),(1,1),W_init=w_init,b_init=None)(nn)
    f=MaxPool2d((2,2),(2,2))(f)
    g=Conv2d(df_dim//4,(1,1),(1,1),W_init=w_init,b_init=None)(nn)
    h=Conv2d(df_dim,(1,1),(1,1),W_init=w_init,b_init=None)(nn)
    h=MaxPool2d((2,2),(2,2))(h)
    s = tf.matmul(Reshape(shape=[g.shape[0],-1,g.shape[1]])(g), Reshape(shape=[f.shape[0],-1,f.shape[1]])(f), transpose_b=True)
    beta=tf.nn.softmax(s)
    o=tf.matmul(beta,Reshape(shape=[h.shape[0],-1,h.shape[1]])(h))
    o=Reshape(shape=[nn.shape[0],nn.shape[1],nn.shape[2],gf_dim])(o)
    o=Conv2d(df_dim*2,(1,1),(1,1),W_init=w_init,b_init=None)(o)
    nn=nn+gamma_init*o
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=Conv2d(df_dim*4,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=nn+n1;
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=Conv2d(df_dim*8,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=nn+n1;
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=Conv2d(df_dim*8,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=nn+n1;
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(nn)
    n1=Conv2d(df_dim*16,(5,5),(2,2),W_init=w_init,b_init=None)(n1);
    nn=nn+n1;
    n1=Conv2d(df_dim*16,(5,5),(2,2),W_init=w_init,b_init=None)(nn);
    n1=BatchNorm2d(decay=0.9,act=tf.nn.relu,gamma_init=gamma_init,name=None)(n1)
    nn=nn+n1;
    nn=tf.reduce_sum(nn,axis=[1,2])
    nn=Dense(n_units=1,W_init=w_init,act=tf.identity)(nn)
    return tl.models.Model(inputs=ni,outputs=nn,name='discriminator')
