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
We require $`\mathbf{r}(s) = (x(s), y(s)) \in [C^2[0, 1]]^2`$ .
We also have a time parametrization of the path given by 
```math
(x(t), y(t)) = (x(s(t)), y(s(t)))
```
The speed along the path $`v \in C^1[0, 1]`$ is defined as
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
& (x, y) \in \mathcal{F}_{xy}
\end{align*}
```
where $`v_{\max}`$ is the maximum speed of the car, $`a_{\min}`$ is the maximum braking deceleration, $`a_{\max}`$ is the maximum positive acceleration, and $`\mu`$ is the friction coefficient.

## Finite Difference Approach
To implement the finite difference method, first we need to discretize the path $`(x(s), y(s))`$ and the speed $`v(s)`$. For any $N \in \mathbb{N}$, define $h:=\frac{1}{N}$. Define the discrete path at $(n+1)$ positions as
```math
(x_0,y_0)\to(x_1, y_1)\to...\to(x_N, y_N)
```
such that $(x_n, y_n) \in \mathcal{F}_{xy}$.
Define the discrete speed as $v_0\to v_1\to...\to v_n$ where $v_0$ is the initial speed entering the track.
Notice that we will use vector notations for the discrete path and speed:
```math
\mathbf{x} = (x_0, x_1, ..., x_N), \quad \mathbf{y} = (y_0, y_1, ..., y_N),  \quad \mathbf{v} = (v_0, v_1, ..., v_N)
```
Now we want to implement numerical differentiation on the discrete path and speed. Let $f \in C^2[0, 1]$, consider the following Taylor expansions of $f(x)$:
```math
\begin{align*}
f(x-2h) &= f(x) - f'(x)2h + \frac{f''(x)}{2}4h^2 + O(h^2)\\
f(x-h) &= f(x) - f'(x)h + \frac{f''(x)}{2}h^2 + O(h^2)\\
f(x+h) &= f(x) + f'(x)h + \frac{f''(x)}{2}h^2 + O(h^2)\\
f(x+2h) &= f(x) + f'(x)2h + \frac{f''(x)}{2}4h^2 + O(h^2)
\end{align*}
```
We derive the finite difference approximations with error $O(h)$ as the following:
```math
\begin{align*}
f'(x) &= \begin{cases}
(f(x+h) - f(x-h))/2h \\
(-3f(x) + 4f(x+h) -f(x+2h))/2h \\
(f(x-2h) -4f(x-h) +3f(x))/2h
\end{cases}\\
f''(x) &= \begin{cases}
(f(x-h) -2f(x) +f(x+h))/h^2 \\
(2f(x) - 5f(x+h) +4f(x+2h) -f(x+3h))/h^2 \\
(-f(x-3h) + 4f(x-2h) -5f(x-h) +2f(x))/h^2
\end{cases}\\
\end{align*}
```
Define the 1st finite derivative matrix $\mathbf{D}^{[1]}_h \in \mathbb{R}^{(N+1)\times(N+1)} $:
```math
\mathbf{D}^{[1]}_h = \frac{1}{2h}\begin{bmatrix}
-3 & 4 & -1 & 0 &\cdots & 0 & 0 & 0\\
-1 & 0 &  1 & 0 &\cdots & 0 & 0 & 0\\
0 & -1 &  0 & 1 &\cdots & 0 & 0 & 0\\
\vdots & \vdots & \vdots & \vdots &  \cdots & \vdots & \vdots & \vdots & \\
0 & 0 & 0 & 0 &\cdots & -1 & 0 & 1\\
0 & 0 & 0 & 0 &\cdots & 1 & -4 & 3\\
\end{bmatrix}
```

Define the 2nd finite derivative matrix $\mathbf{D}^{[2]}_h \in \mathbb{R}^{(N+1)\times(N+1)} $:
```math
\mathbf{D}^{[2]}_h = \frac{1}{h^2}\begin{bmatrix}
2 & -5 & 4 & -1 &\cdots & 0 & 0 & 0 & 0\\
1 & -2 &  1 & 0 &\cdots & 0 & 0 & 0 & 0\\
0 & 1 &  -2 & 1 &\cdots & 0 & 0 & 0 & 0\\
\vdots & \vdots & \vdots & \vdots &  \cdots & \vdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & 0 &\cdots & 0 & 1 & -2 & 1\\
0 & 0 & 0 & 0 &\cdots & -1 & 4 & -5 & 2\\
\end{bmatrix}
```
For $i \in \{0, 1, ..., N\}$ and $j\in\{1, 2\}$, let $\delta^{[j]}_i x = [\mathbf{D}^{[j]} \mathbf{x}]_i$, similarly for $\mathbf{y}, \mathbf{v}$.
Finally, we can construct the minimization problem on the function spaces into the problem on the Euclidean space as the following:
```math
\begin{align*}
\min_{\mathbf{x}, \mathbf{y}, \mathbf{v} \in \mathbb{R}^{(N+1)}} & \sum_{i=0}^{N} \frac{1}{d_i v_i } - \frac{1}{2}(\frac{1}{v_0 d_0} + \frac{1}{v_N d_N})\\
\textit{s.t.} & \quad d_i = ((\delta_i^{[1]}x)^2 + (\delta_i^{[1]}y)^{2})^{-0.5} \\
 0 & \leq v_i \leq v_{\max} \\
& a_{\min} \leq  d_i v_i \delta_i^{[1]}v \leq a_{\max} \\
& d_i^3 v_i^2 |\delta_i^{[2]}x \delta_i^{[1]}y - \delta_i^{[2]}y \delta_i^{[1]}x| \leq \mu g \\
& (x_i, y_i) \in \mathcal{F}_{xy}
\end{align*} 
```