# Homework Task 4: Cartpole

A task consists of a pole on a cart (weirdly named *Cartpole*). It is our task to provide a optimal action for every state a cartpole is in. 

A state consists of four variables:

- $x$, which is cart's *position*
- $v$, which is cart's *velocity*
- $\theta$, which is pole's *angular position*
- $\dot{\theta}$, which is pole's *angular velocity*

The equations of motion are represented using state space representation:

<p align=center>
 
  $\dot{x_0} = x_1$
  
  $\dot{x_1} \equiv F = \frac{4f - m\sin{(x_2)}(3g\cos{(x_3)} - 4lx_3^{2})}{4(m + M) - 3m\cos^{2}{(x_2)}}$
  
  $\dot{x_2} = x_3$

  $\dot{x_3} \equiv G = \frac{(m + M)g\sin{(x_2)} - \cos{(x_2)}(f + ml\sin{(x_2)}x_3^{2})}{l(\frac{4}{3}(m + M) - m\cos^{2}{(x_2)})}$ 

</p>

The whole system is discretized using *forward differentiation* (*Euler1*) method. The state space equations then become 

<p align=center>

$x_0(k + 1) = x_0(k) + Tx_1(k)$

$x_1(k + 1) = x_1(k) + TF(k)$

$x_2(k + 1) = x_2(k) + Tx_3(k)$

$x_3(k + 1) = x_3(k) + TG(k)$

</p>  
