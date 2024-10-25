# Racing Line Optimization
Author: Sanyang Liu
## Index
- [About the project](#about-the-project)
- [Mathematics Background](#mathematics-background)
- [Finite Difference Approach](#finite-difference-approach)
## About the project 
This project focuses on optimizing the racing line for Formula 1 cars on a track. The goal is to determine the optimal path and speed that minimizes lap times. We primarily use the finite difference method and physics-informed neural networks for our analysis.\
The original idea is from the author's Year 1 individual project on Calculus of Variation at Imperial College London, where the optimization problem was framed using variational principles and solved using the finite difference method.\
## Mathematics background
In this project, we focus exclusively on a 2D planar representation of the track, which simplifies the problem by ignoring vertical height differences and the effects of elevation changes on the car’s dynamics. Then the positions of the car can be represented as $(x, y)$.\
Define the track $\mathcal{F}_{xy}$ as the feasible domain of the position $(x, y)$ of the car. In the track, define the initial position $(x_0, y_0) \in \mathcal{F}_{xy}$ and finish line $\Gamma\subset\mathcal{F}_{xy}$.\
Define the path via the arc-length parametrization in the given track by $\mathbf{r}: [0, 1] \to \mathcal{F}_{xy}$ such that
$$\mathbf{r}(0) = (x_0, y_0)  \quad \text{and}\quad \mathbf{r}(1) \in \Gamma$$ We require $\mathbf{r}(s) = (x(s), y(s)) \in [C^2([0, 1])]^2$.\
We also have the time parametrization of the path $$(x(t), y(t)) = (x(s(t)), y(s(t)))$$ Then define the speed on the path $v\in C^1([0, 1])$ by
$$v(s) = v(s(t)) = \frac{ds}{dt}\sqrt{(\frac{dx}{ds})^2 +(\frac{dy}{ds})^2 }$$
Then the total time along the path is given by the integral which is a functional of $(x, y, v)$:
$$ \mathcal{T} := \int_{[0, 1]} \frac{\sqrt{{x'}^2 +{y'}^2 }}{v} ds $$
We finally formulate the following optimization problem:
$$\begin{align*} 
\min_{x,y,v} & \int_{[0, 1]} \frac{\sqrt{{x'}^2 +{y'}^2 }}{v} ds \\
\textit{s.t.}  \quad 0 & \leq v\leq v_{\max}\\
& a_{\min} \leq \frac{v v'}{\sqrt{{x'}^2 +{y'}^2 }} \leq a_{\max}\\
& v^2 \frac{|x'' y' - y''x'|}{({x'}^2 + {y'}^2)^{\frac{3}{2}}} \leq \mu g \\
& (x, y) \in \mathcal{F}_{xy} \\
\end{align*}$$
where $v_{max}$ is the maximum speed of the car, $a_{min}$ is the maximum brake deceleration, $a_{max}$ is the maximum positive acceleration and $\mu$ is the friction coefficient.

## Finite Difference Approach