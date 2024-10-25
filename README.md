# Racing Line Optimization
**Author:** Sanyang Liu

## Index
- [About the Project](#about-the-project)
- [Mathematics Background](#mathematics-background)
- [Finite Difference Approach](#finite-difference-approach)

## About the Project
This project focuses on optimizing the racing line for Formula 1 cars on a track. The goal is to determine the optimal path and speed that minimizes lap times. We primarily use the finite difference method and physics-informed neural networks for our analysis. The original idea stems from my Year 1 individual project on Calculus of Variations at Imperial College London, where the optimization problem was framed using variational principles and solved with the finite difference method.

## Mathematics Background
In this project, we concentrate exclusively on a 2D planar representation of the track, simplifying the problem by ignoring vertical height differences and the effects of elevation changes on the car’s dynamics. The positions of the car can be represented as $`(x, y)`$.

We define the track $`\mathcal{F}_{xy}`$ as the feasible domain for the car's position $`(x, y)`$ . Within the track, we identify the initial position $`(x_0, y_0) \in \mathcal{F}_{xy}`$ and the finish line $`\Gamma \subset \mathcal{F}_{xy}`$.
The path is defined via arc-length parametrization in the given track by $`\mathbf{r}: [0, 1] \to \mathcal{F}_{xy}`$ , such that
```math
\mathbf{r}(0) = (x_0, y_0) \quad \text{and} \quad \mathbf{r}(1) \in \Gamma
```
We require $`\mathbf{r}(s) = (x(s), y(s)) \in [C^2([0, 1])]^2`$ .
We also have a time parametrization of the path given by 
```math
(x(t), y(t)) = (x(s(t)), y(s(t)))
```
The speed along the path $`v \in C^1([0, 1])`$ is defined as
```math
v(s) = v(s(t)) = \frac{ds}{dt} \sqrt{\left(\frac{dx}{ds}\right)^2 + \left(\frac{dy}{ds}\right)^2 }
```
The total time along the path is given by the integral, which serves as a functional of $`(x, y, v)`$ :
```math
\mathcal{T} := \int_{[0, 1]} \frac{\sqrt{{x'}^2 + {y'}^2}}{v} ds
```
We finally formulate the following optimization problem:
```math
\begin{align*}
\min_{x,y,v} & \int_{[0, 1]} \frac{\sqrt{{x'}^2 + {y'}^2}}{v} ds \\
\textit{s.t.}  \quad 0 & \leq v \leq v_{\max} \\
& a_{\min} \leq \frac{v v'}{\sqrt{{x'}^2 + {y'}^2}} \leq a_{\max} \\
& v^2 \frac{|x'' y' - y'' x'|}{({x'}^2 + {y'}^2)^{\frac{3}{2}}} \leq \mu g \\
& (x, y) \in \mathcal{F}_{xy} \\
\end{align*}
```
where $`v_{\max}`$ is the maximum speed of the car, $`a_{\min}`$ is the maximum braking deceleration, $`a_{\max}`$ is the maximum positive acceleration, and $`\mu`$ is the friction coefficient.

## Finite Difference Approach
