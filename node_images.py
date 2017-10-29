import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
patch1 = matplotlib.patches.Circle(
    [0.5,0.5],0.05
)
patch2 = matplotlib.patches.Rectangle(
    [0.3,0.3],0.4, 0.4, alpha=0.5, 
    fill=False, edgecolor='black',
    linestyle = '--'
)
arrow1 = matplotlib.patches.Arrow(
    0, 0.5,0.45,0, width=0.05,
    color='black'
)
arrow2 = matplotlib.patches.Arrow(
    0.55, 0.5,0.45,0, width=0.05,
    color='black'
)
line1 = matplotlib.lines.Line2D(
    [0.5,0.5], [0,0.45],
    linestyle='--', color='black'
)
text1 = matplotlib.text.Text(
    0, 0.45, '$n_{A0}$\n$V_0$\n$U_A=0$'
)
text2 = matplotlib.text.Text(
    0.8, 0.45, '$n_{A1}$\n$V_1$\n$U_{A1}$'
)
for artist in [
    patch1,patch2,arrow1,arrow2,
    line1,text1,text2
]:
    ax.add_artist(artist)
ax.set_frame_on(False)
ax.set_axis_off()
ax.set_aspect(1.0)

fig.


fig = plt.figure()
ax = fig.add_subplot(111)
patch1 = matplotlib.patches.Circle(
    [0.5,0.5],0.05
)
patch2 = matplotlib.patches.Rectangle(
    [0.3,0.3],0.4, 0.4, alpha=0.5, 
    fill=False, edgecolor='black',
    linestyle = '--'
)
arrow1 = matplotlib.patches.Arrow(
    0, 0.5,0.45,0, width=0.05,
    color='black'
)
arrow2 = matplotlib.patches.Arrow(
    0.55, 0.5,0.45,0, width=0.05,
    color='black'
)
arrow3 = matplotlib.patches.Arrow(
    0.5, 0.0, 0,0.45, width=0.05,
    color='black'
)
text1 = matplotlib.text.Text(
    0, 0.45, '$n_{A0}$\n$V_0$\n$U_A=0$'
)
text2 = matplotlib.text.Text(
    0.8, 0.45, '$n_{A1}$\n$V_1$\n$U_{A1}$'
)
text3 = matplotlib.text.Text(
    0.55, 0.1, '$n_{Ar}$\n$V_r$'
)
for artist in [
    patch1,patch2,arrow1,arrow2,
    arrow3,text1,text2,text3
]:
    ax.add_artist(artist)
ax.set_frame_on(False)
ax.set_axis_off()
ax.set_aspect(1.0)