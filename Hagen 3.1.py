r"""°°°
# Hagen 3.1 [10]
Ein Röstgas für die Herstellung von Schwefelsäure nach dem Kontaktverfahren hat die folgende Zusammensetzung in Vol.-%:

  comp | Vol%
     - | - 
 $SO_2$ | 7,8%
 $O_2$ | 10,8%
 $N_2$ | 81,4%

Der Gleichgewichtsumsatz bei 500°C und 1 bar beträgt 0,96.

$SO_2 + \frac{1}{2} O_2 
{_{\leftarrow}^{\rightarrow}} SO_3$

Berechne die Zusammensetzung des Gasgemisches nach Erreichen des Gleich-gewichts.
°°°"""
# |%%--%%| <qvnPnFeR0U|KgXQqq7zcT>
r"""°°°
## Lösung
Umsatz, auf der Basis des Schwefeldioxids: 

$U_k =0,96= 1 - \frac{x_{SO_2}\dot{n}}{x_{0, SO_2}\dot{n_0}} $

Aus den Billanzen in einem geschlossenen Rührkessel ($n_i = n_{0, i} + \sum_j{\nu_{ij} \xi_j}$) entsteht ein System von 6 unabhängigen Gleichungen:

| no. | Gl.  |
| - | - |
| 1 | $\rlap{x_{SO_2}\dot{n} = x_{0, SO_2}\dot{n_0} + \nu_{SO_2}\xi_1}$ |
| 2 | $\rlap{x_{O_2}\dot{n} = x_{0, O_2}\dot{n_0} + \nu_{O_2}\xi_1}$ |
| 3 | $\rlap{x_{N_2}\dot{n} = x_{0, N_2}\dot{n_0} + \nu_{N_2}\xi_1}$ |
| 4 | $\rlap{x_{SO_3}\dot{n} = x_{0, SO_3}\dot{n_0} + \nu_{SO_3}\xi_1}$ |
| 5 | $\rlap{1 = x_{SO_2} + x_{O_2} + x_{N_2} + x_{SO_3}}$ |
| 6 | $\rlap{U_k =0,96= 1 - \frac{x_{SO_2}\dot{n}}{x_{0, SO_2}\dot{n_0}}}$ |

Da in diesem System noch 7 Veränderlichen zu bestimmen stehen ($x_{SO_2}$, $x_{O_2}$, $x_{N_2}$, $x_{SO_3}$, $\dot{n}$, $\dot{n_0}$, $\xi_1$), erfolgt dessen Lösung durch die Verringerung der Anzahl an Variablen. Zwei Gruppen weisen sich als geeignet auf, um eine Rationalisierung durchzuführen: 
* Reaktionslaufzahl pro Einheit zulaufender Stoffmenge  $\left(\frac{\xi_1}{\dot{n_0}}\right)$
* Auslaufende Stoffmenge pro Einheit zulaufender Stoffmenge $\left(\frac{\dot{n}}{\dot{n_0}}\right)$

Das System wird folgendermaßen umgesetzt:

| no. | Gl.  |
| - | - |
| 1 | $\rlap{x_{SO_2}\left(\frac{\dot{n}}{\dot{n_0}}\right) = x_{0, SO_2} + \nu_{SO_2}\left(\frac{\xi_1}{\dot{n_0}}\right)}$ |
| 2 | $\rlap{x_{O_2}\left(\frac{\dot{n}}{\dot{n_0}}\right) = x_{0, O_2} + \nu_{O_2}\left(\frac{\xi_1}{\dot{n_0}}\right)}$ |
| 3 | $\rlap{x_{N_2}\left(\frac{\dot{n}}{\dot{n_0}}\right) = x_{0, N_2} + \nu_{N_2}\left(\frac{\xi_1}{\dot{n_0}}\right)}$ |
| 4 | $\rlap{x_{SO_3}\left(\frac{\dot{n}}{\dot{n_0}}\right) = x_{0, SO_3} + \nu_{SO_3}\left(\frac{\xi_1}{\dot{n_0}}\right)}$ |
| 5 | $\rlap{1 = x_{SO_2} + x_{O_2} + x_{N_2} + x_{SO_3}}$ |
| 6 | $\rlap{U_k =0,96= 1 - \frac{x_{SO_2}}{x_{0, SO_2}}\left(\frac{\dot{n}}{\dot{n_0}}\right)}$ |

Mit 6 Veränderlichen und umsoviel Gleichungen lässt sich eine Lösung finden. Gleichungen 1 zusammen mit 6 ermöglichen die sofortige Rechnung der Reaktionslaufzahl-Proportion, wonach die Auslaufstoffmengen-Proportion durch die Summe von Gleichungen 1 bis 4 und anhand der Gleichung 5 berechnet wird: 

* Aus 1 mit 6: $\left(\frac{\xi_1}{\dot{n_0}}\right) = \frac{-U_{SO_2}}{\nu_{SO_2}}\times x_{0, SO_2} =  \frac{-0,96}{-1}\times 0,078$

* Aus 1 bis 5: $(1)\left(\frac{\dot{n}}{\dot{n_0}}\right)  = 1 + \left(\sum_i{\nu_{i1}} \right)\times \left(\frac{\xi_1}{\dot{n_0}}\right)$

$\left(\frac{\dot{n}}{\dot{n_0}}\right) = 1 + \left(\sum_i{\nu_{i1}} \right)\times \left(\frac{-U_{SO_2}}{\nu_{SO_2}}\times x_{0, SO_2} \right)=1 + \left(-1-1/2+1 \right)\times \left(\frac{-0,96}{-1}\times 0,078 \right)$

Somit kann jeder Molenbruch berechnet werden, unabhängig von den Zulaufs- und Auslaufströmen:

$x_i = \frac{x_{0, i}+\nu_i \left(\frac{\xi_1}{\dot{n_0}}\right)}{ 1 + \left(\sum_i{\nu_{i1}} \right)\times \left(\frac{\xi_1}{\dot{n_0}}\right)}$
°°°"""
# |%%--%%| <KgXQqq7zcT|8Wmr0OJjet>

x0so2 = 0.078
x0n2 = 0.814
x0o2 = 0.108
x0so3 = 0.0

nuso2 = -1.0
nun2 = 0.0
nuo2 = -1.0/2.0
nuso3 = 1.0

uk = 0.96

xi_durch_n0 = -uk/nuso2 * x0so2
n_durch_n0 = 1 + sum([nuso2, nun2, nuo2, nuso3]) * xi_durch_n0
xso2 = (x0so2 + nuso2 * xi_durch_n0)/n_durch_n0
xn2 = (x0n2 + nun2 * xi_durch_n0)/n_durch_n0
xo2 = (x0o2 + nuo2 * xi_durch_n0)/n_durch_n0
xso3 = (x0so3 + nuso3 * xi_durch_n0)/n_durch_n0

for var in [
    'xi_durch_n0', 'n_durch_n0', 'xso2', 'xn2', 'xo2', 'xso3'
    ]:
    print var + ': ' + '{0:g}'.format(locals()[var])
