from EduNLP import get_pretrained_i2v
from EduNLP.Vector.t2v import PRETRAINED_MODELS
from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V
from pathlib import PurePath


item = {
  "stem": r"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
  "options": ["$p_1=p_2$", "$p_1=p_3$", "$p_2=p_3$", "$p_1=p_2+p_3$"]
}
i2v = get_pretrained_i2v("d2v_eng_256")
print(i2v(item))
