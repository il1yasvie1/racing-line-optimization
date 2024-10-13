# Racing Line Optimization

Suppose we reparametrize the curve

$$v(s) = v(s(t)) = \frac{ds}{dt}\sqrt{(\frac{dx}{ds})^2 +(\frac{dy}{ds})^2 }$$

$$\begin{align*} 
\min_{x,y,v} & \int_{[0, 1]} \frac{1}{v(s)} ds \\
\textit{s.t.} &  \sqrt{(\frac{dx}{ds})^2 +(\frac{dy}{ds})^2 }= 1 \\
& 0 \leq v\leq v_{\max}\\
& a_{\min} \leq v\frac{dv}{ds} \leq a_{\max}\\
& v^2 \sqrt{(\frac{d^2 x}{ds^2})^2 +(\frac{d^2y}{ds^2})^2 } \leq \frac{\mu g}{M} \\
& (x, y) \in \Omega \\
\end{align*}$$