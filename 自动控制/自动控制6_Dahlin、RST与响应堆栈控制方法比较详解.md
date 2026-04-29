# Examen d’automatique 6 — 详细讲解与解答

说明：本文件针对 `MinerU_Examen d’automatique6__20251127111709.md`（一阶滞后含时延的 Dahlin vs RST vs 响应堆栈比较）撰写。每问含题意解析与解题步骤/结果，统一保留三位小数。

目录
- 1 Partie 1：Dahlin 近似（8 分，50 分钟）
- 2 Partie 2：RST 合成（7 分，50 分钟）
- 3 Partie 3：Réponse pile（5 分，20 分钟）

---

## 1 Partie 1：Dahlin 近似

对象（ZOH 离散后）：
$$
G(p)=\frac{g}{1+\tau p}e^{-Tp}\;\Rightarrow\;G(z)=\frac{g(1-a)}{1-a z^{-1}}\,z^{-d},\quad a=e^{-T_e/\tau},\;T=(d-1)T_e.
$$

2) 题意解析（由差分式到 C(z)）
- 已知控制器差分：
$$
\boxed{\;u_k=\alpha u_{k-1}+(1-\alpha)u_{k-d}+\beta\,\varepsilon_k - b\beta\,\varepsilon_{k-1}\;}
$$
- $z$ 域：
$$
(1-\alpha z^{-1}-(1-\alpha)z^{-d})U(z)=\beta(1-b z^{-1})\,\varepsilon(z)\;\Rightarrow\;
\boxed{\;C(z)=\frac{U}{\varepsilon}=\frac{\beta(1-b z^{-1})}{1-\alpha z^{-1}-(1-\alpha)z^{-d}}\;}
$$

3) 题意解析（令 b=a，闭环 BF）
- $L(z)=C(z)G(z)=\dfrac{\beta(1-b z^{-1})}{1-\alpha z^{-1}-(1-\alpha)z^{-d}}\cdot\dfrac{g(1-a)}{1-a z^{-1}}z^{-d}$，当 $b=a$ 抵消后：
$$
L(z)=\frac{\beta g(1-a)\,z^{-d}}{1-\alpha z^{-1}-(1-\alpha)z^{-d}}.
$$
- 闭环：
$$
BF(z)=\frac{L(z)}{1+L(z)}=\frac{\beta g(1-a)\,z^{-d}}{1-\alpha z^{-1}+[\beta g(1-a)-(1-\alpha)]z^{-d}}.
$$
- 若取
$$\boxed{\;\beta=\frac{1-\alpha}{g(1-a)}\;},$$
则
$$\boxed{\;BF(z)=\frac{(1-\alpha)\,z^{-d}}{1-\alpha z^{-1}}\;}\quad\text{（一阶延迟 + 纯延时）。}
$$

5) 题意解析（静差）
- $BF(z)$ 有单位增益（$z=1$）且分母含 $1-\alpha z^{-1}$，对阶跃 $W(z)=1/(1-z^{-1})$：
$$
\lim_{z\to1}(1-z^{-1})\,BF(z)\,W(z)=1\Rightarrow\boxed{\;\varepsilon(\infty)=0\;}
$$

6) 题意解析（稳定域示例）
- 以 $d=2$：$\mathrm{Den}[C]=1-\alpha z^{-1}-(1-\alpha)z^{-2}=(1-z^{-1})(1-(\alpha-1)z^{-1})$，需 $|\alpha-1|<1\Rightarrow 0<\alpha<2$。
- 以 $d=3$：$\mathrm{Den}[C]=(1-z^{-1})(1+z^{-1}+(1-\alpha)z^{-2})$，Jury 给 $0<\alpha<1.5$（题面示例）。

8) 题意解析（\alpha\to0 的极限）
- $BF(z)\to z^{-d}$（延迟纯采样周期 $d$），符合理想“最小时间”但无整形。

9) 题意解析（扰动通道 $F_{PY}$ 在 \alpha=0、b=a、\beta 取上式）
- 推导：
$$
F_{PY}(z)=\frac{G(z)}{1+C(z)G(z)}=\frac{\tfrac{g(1-a)}{1-a z^{-1}}\,z^{-d}}{1+\tfrac{(1-\alpha)}{(1-a)}\cdot\tfrac{z^{-d}}{1-\alpha z^{-1}-(1-\alpha)z^{-d}}}\Bigg|_{\alpha=0}
=\frac{g(1-a)}{1-a z^{-1}}\,z^{-d}(1-z^{-d}).
$$
- 结论：其极点仍由 $1-a z^{-1}$ 决定（与开环一致），拒扰动态与开环相同。

10) 题意解析（扰动阶跃下的最终值）
- $P(z)=1/(1-z^{-1})$：
$$
\lim_{z\to1}(1-z^{-1})Y(z)=\lim_{z\to1}\frac{(1-z^{-1})}{1-a z^{-1}}\,g(1-a)\,z^{-d}(1-z^{-d})=0.
$$
- 结论：渐近消除（因为 $(1-z^{-d})\to0$）。

---

## 2 Partie 2：RST 合成

对象分解：
$$
G(z)=\frac{B}{A}=\frac{B^{+}B^{-}}{(1-z^{-1})^{m}A^{+}A^{-}},\quad B^{+}=1-a z^{-1},\;B^{-}=z^{-d},\;A^{-}=1.
$$
2) 题意解析（扰动通道）
$$
\boxed{\;F_{PY}(z)=\frac{B\,S}{A\,S+B\,R}\;}
$$
3) 题意解析（$S=(1-z^{-1})S_1$ 的意义）
- 令 $S$ 含 $(1-z^{-1})$，可对阶跃扰动实现零静差（使 $F_{PY}$ 成多项式）。

4) 题意解析（丢番图式）
- 目标：$A S + B R = 1$（使 $F_{PY}$ 为纯多项式）。
- 展开：
$$
(1-a z^{-1})(1-z^{-1})S_1(z)+(1-a)z^{-d}R(z)=1.
$$
5) 题意解析（$d=2$ 的阶次选择与求解）
- 取 $\deg R=1$，$R=r_0+r_1 z^{-1}$；$\deg S_1=1$，$S_1=s_0+s_1 z^{-1}$。
- 展开并匹配常数与 $z^{-1},z^{-2}$ 系数，得：
$$
\boxed{\;s_0=1,\;s_1=a+1,\;r_0=-\frac{a+(a+1)^2}{1-a},\;r_1=-\frac{a(a+1)}{1-a}\;}
$$
（题面已给）

7) 题意解析（跟踪通道）
- 令 $F_{WY}(z)=\dfrac{B T}{A S + B R}=B T=(1-a)z^{-d}\,T$。
- 若目标 $F_{WY}(z)=z^{-d}$：
$$\boxed{\;T=\frac{1}{1-a}\;}
$$

---

## 3 Partie 3：Réponse pile

1) 题意解析（d=2 的堆栈合成）
- 目标：
$$
(1-z^{-1})K(z)+g(1-a)z^{-2}L(z)=1.
$$
- 阶次：$K=k_0+k_1 z^{-1}$，$L=l_0$（题面示例）。
- 求得：$k_0=1,\;k_1=1,\;l_0=\dfrac{1}{g(1-a)}$，
- 校正器：
$$\boxed{\;C(z)=\frac{(1-a z^{-1})\,l_0}{(1-z^{-1})(1+z^{-1})}\;}
$$
2) 题意解析（不稳定性）
- 分母含 $(1+z^{-1})$，在 $z=-1$ 处有极点（单位圆上，非稳定）。故“不稳定”。

3) 题意解析（与 Dahlin 的关系）
- 当取 $\alpha=0$ 时，Dahlin 的 $C(z)$ 与此“堆栈”在特例上等价（避免 $(1+z^{-1})$ 的情形），一般 Dahlin 是堆栈的稳定近似。

---

# 总结与建议
- Dahlin：参数 $\alpha,\beta,b$ 可使闭环成为“延迟 + 一阶”形态，跟踪与稳定性较易把控，但扰动拒斥能力受开环主极点限制。
- RST：通过 $AS+BR=1$ 可将扰动通道做成纯多项式（型别控制），并用 $T$ 独立设定跟踪增益与延迟，整体可获得更好的拒扰与跟踪折中。
- 堆栈：是理想“最小时间”策略的原型，但在某些结构下会引入 $z=-1$ 极点导致不稳定；Dahlin 可视为其稳定化近似。

如需，我可以把以上各段的差分方程（$u_k$ 关于过去 $u,e$ 的显式形式）整理到附录，并给一个可直接仿真的最简数值示例。