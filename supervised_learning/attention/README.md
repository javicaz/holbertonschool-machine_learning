Attention

Resources:
Read or watch:

Attention Model Intuition
Attention Model
How Transformers work in deep learning and NLP: an intuitive introduction
Transformers
Bert, GPT : The Illustrated GPT-2 - Visualizing Transformer Language Models
SQuAD
Glue
Self supervised learning
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is the attention mechanism?
How to apply attention to RNNs
What is a transformer?
How to create an encoder-decoder transformer model
What is GPT?
What is BERT?
What is self-supervised learning?
How to use BERT for specific NLP tasks
What is SQuAD? GLUE?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)
Your files will be executed with numpy (version 1.19.2) and tensorflow (version 2.6)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
All of your files must be executable
A README.md file, at the root of the folder of the project, is mandatory
Your code should follow the pycodestyle style (version 2.6)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise stated, you cannot import any module except import tensorflow as tf
Quiz questions
Great! You've completed the quiz successfully! Keep going! (Show quiz)
Tasks
0. RNN Encoder
mandatory
Resources:

Encoder-Decoder Architecture
Create a class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation:

Class constructor def __init__(self, vocab, embedding, units, batch):
vocab is an integer representing the size of the input vocabulary
embedding is an integer representing the dimensionality of the embedding vector
units is an integer representing the number of hidden units in the RNN cell
batch is an integer representing the batch size
Sets the following public instance attributes:
batch - the batch size
units - the number of hidden units in the RNN cell
embedding - a keras Embedding layer that converts words from the vocabulary into an embedding vector
gru - a keras GRU layer with units units
Should return both the full sequence of outputs as well as the last hidden state
Recurrent weights should be initialized with glorot_uniform
Public instance method def initialize_hidden_state(self):
Initializes the hidden states for the RNN cell to a tensor of zeros
Returns: a tensor of shape (batch, units)containing the initialized hidden states
Public instance method def call(self, x, initial):
x is a tensor of shape (batch, input_seq_len) containing the input to the encoder layer as word indices within the vocabulary
initial is a tensor of shape (batch, units) containing the initial hidden state
Returns: outputs, hidden
outputs is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
hidden is a tensor of shape (batch, units) containing the last hidden state of the encoder
$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNEncoder = __import__('0-rnn_encoder').RNNEncoder

encoder = RNNEncoder(1024, 128, 256, 32)
print(encoder.batch)
print(encoder.units)
print(type(encoder.embedding))
print(type(encoder.gru))

initial = encoder.initialize_hidden_state()
print(initial)
x = tf.convert_to_tensor(np.random.choice(1024, 320).reshape((32, 10)))
outputs, hidden = encoder(x, initial)
print(outputs)
print(hidden)
$ ./0-main.py
32
256
<class 'keras.layers.embeddings.Embedding'>
<class 'keras.layers.recurrent_v2.GRU'>
tf.Tensor(
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]], shape=(32, 256), dtype=float32)
tf.Tensor(
[[[-1.69446021e-02 -1.13353292e-02 -2.65584420e-03 ... -3.37935193e-03
   -2.75800157e-05  1.98357317e-04]
  [-1.13625592e-02 -1.94185833e-03 -3.32558411e-03 ...  4.02807118e-03
    4.85287188e-03  1.18722406e-03]
  [ 5.28527889e-04 -9.85167269e-03 -1.58155840e-02 ...  1.05810566e-02
   -1.99264521e-03  7.62981270e-03]
  ...
  [ 9.51008778e-03 -1.03162089e-03  2.20384938e-03 ...  1.52750686e-03
   -1.62444077e-02 -8.31484888e-03]
  [ 1.48318764e-02 -2.00394867e-03  6.41316222e-03 ...  5.50597999e-03
   -1.19308010e-04 -3.61553114e-03]
  [-1.49735017e-03 -5.78260841e-03 -1.40024740e-02 ... -3.01877572e-03
    1.07422806e-02 -7.24770967e-03]]

 [[-9.22318734e-03 -2.71162950e-03  1.31640001e-03 ...  3.40229890e-04
   -1.63394224e-03  1.04985349e-02]
  [-2.97956262e-03  5.41196950e-03 -8.25727545e-03 ...  4.59889043e-03
   -4.05747909e-03  4.97913826e-03]
  [-1.78691000e-04  2.03597359e-03 -9.94522125e-04 ...  6.65585976e-04
    1.67084802e-02 -1.73955527e-03]
  ...
  [-1.97272259e-03 -3.78618273e-03 -5.36715426e-03 ... -1.14764227e-03
    2.56696506e-03 -1.39833083e-02]
  [-6.32762257e-03 -7.70640047e-03 -3.03669367e-03 ...  1.95274595e-03
   -4.21465142e-03 -1.83490496e-02]
  [-1.14143295e-02 -5.21823578e-03  3.37029272e-03 ...  8.05946440e-03
   -6.54672086e-03 -1.20249605e-02]]

 [[-7.06690829e-03  4.30702837e-03 -8.89914017e-03 ... -1.03690904e-02
   -4.20188066e-03 -1.32986046e-02]
  [ 7.43784290e-03  1.01290073e-03  6.43281359e-03 ... -7.27578485e-03
    1.88174527e-02 -9.70994495e-03]
  [ 8.45354795e-03  7.64563913e-03  4.63636033e-03 ... -1.04573164e-02
    1.70352645e-02 -2.92225275e-04]
  ...
  [ 7.82680511e-03  3.58770462e-03  3.88183817e-03 ... -7.41777522e-03
   -1.04718814e-02  1.52943749e-02]
  [-8.80663097e-03  1.03175137e-02  5.98307792e-03 ...  1.27293067e-02
   -1.36418082e-02  1.45526212e-02]
  [-6.84839021e-03  4.55516856e-03  1.13999126e-02 ...  1.07124913e-03
   -1.46288686e-02  8.85307789e-03]]

 ...

 [[-1.48272682e-02  9.24249832e-03  3.32123344e-03 ...  1.18100470e-04
    1.18980436e-02 -8.57335888e-03]
  [-6.38324302e-03  3.66323115e-03 -3.78455222e-03 ... -7.34516419e-03
    7.88846798e-03  7.25242961e-03]
  [ 1.02697890e-02  4.16091969e-03 -1.26993854e-03 ... -1.00397030e-02
    5.53037366e-03  6.47463743e-03]
  ...
  [ 9.61682759e-03 -2.62920838e-03 -2.92416289e-03 ... -1.21719157e-02
    3.00088851e-03 -1.96036394e-03]
  [ 5.27984183e-03  1.97624648e-03 -7.73051102e-03 ... -1.11616533e-02
    2.33651861e-03  4.11779899e-03]
  [ 8.93655047e-03  6.54357579e-03  2.08069989e-03 ... -3.76952812e-04
    3.89854866e-03 -4.76602698e-04]]

 [[-2.22250517e-03  9.66395950e-04 -3.68046761e-03 ...  1.84869918e-03
   -1.83799898e-03  8.54290382e-04]
  [-3.29815084e-04 -1.30018359e-02  2.33474653e-03 ...  1.03131821e-02
   -2.61159940e-03  5.50118508e-03]
  [ 1.38921626e-02 -8.67047906e-03  9.75352153e-03 ... -4.74040443e-03
    1.26398704e-03 -2.67816521e-03]
  ...
  [-1.03130778e-02 -2.66719284e-03  3.32104973e-04 ... -1.83578581e-02
    1.53097874e-02 -1.93317072e-03]
  [-1.41187869e-02 -1.35098863e-02 -2.05840362e-04 ... -2.19966210e-02
    3.06172669e-03  4.81538568e-03]
  [-3.31083918e-03 -1.42251169e-02  3.10055475e-05 ... -2.17056870e-02
    4.03413968e-03 -1.07096871e-02]]

 [[-3.37713718e-04 -2.99246376e-03  9.03424248e-03 ... -1.45055854e-03
    8.49891361e-03 -2.18083407e-03]
  [-1.88941285e-02 -5.40235266e-03  1.69082987e-03 ... -2.68692290e-03
    2.63337255e-03  1.35215670e-02]
  [-9.09076817e-03  1.29015744e-03 -3.11447657e-03 ... -7.32218474e-03
    1.80800620e-03  6.74866140e-03]
  ...
  [-4.34762565e-03  2.41552852e-03  6.92352559e-03 ... -2.06366135e-03
    1.44171258e-02  1.19797410e-02]
  [-3.73053714e-04 -2.15956802e-03  1.12224761e-02 ... -1.33350473e-02
    8.17014370e-03  5.56233805e-04]
  [ 2.44038552e-03 -7.95266870e-03  1.52815729e-02 ...  1.49750784e-02
    5.41469455e-03 -2.65267468e-03]]], shape=(32, 10, 256), dtype=float32)
tf.Tensor(
[[-1.49735017e-03 -5.78260841e-03 -1.40024740e-02 ... -3.01877572e-03
   1.07422806e-02 -7.24770967e-03]
 [-1.14143295e-02 -5.21823578e-03  3.37029272e-03 ...  8.05946440e-03
  -6.54672086e-03 -1.20249605e-02]
 [-6.84839021e-03  4.55516856e-03  1.13999126e-02 ...  1.07124913e-03
  -1.46288686e-02  8.85307789e-03]
 ...
 [ 8.93655047e-03  6.54357579e-03  2.08069989e-03 ... -3.76952812e-04
   3.89854866e-03 -4.76602698e-04]
 [-3.31083918e-03 -1.42251169e-02  3.10055475e-05 ... -2.17056870e-02
   4.03413968e-03 -1.07096871e-02]
 [ 2.44038552e-03 -7.95266870e-03  1.52815729e-02 ...  1.49750784e-02
   5.41469455e-03 -2.65267468e-03]], shape=(32, 256), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 0-rnn_encoder.py
 
0/5 pts
1. Self Attention
mandatory
Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on this paper:

Class constructor def __init__(self, units):
units is an integer representing the number of hidden units in the alignment model
Sets the following public instance attributes:
W - a Dense layer with units units, to be applied to the previous decoder hidden state
U - a Dense layer with units units, to be applied to the encoder hidden states
V - a Dense layer with 1 units, to be applied to the tanh of the sum of the outputs of W and U
Public instance method def call(self, s_prev, hidden_states):
s_prev is a tensor of shape (batch, units) containing the previous decoder hidden state
hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
Returns: context, weights
context is a tensor of shape (batch, units) that contains the context vector for the decoder
weights is a tensor of shape (batch, input_seq_len, 1) that contains the attention weights
$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

attention = SelfAttention(256)
print(attention.W)
print(attention.U)
print(attention.V)
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)), dtype='float32')
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)), dtype='float32')
context, weights = attention(s_prev, hidden_states)
print(context)
print(weights)
$ ./1-main.py
<keras.layers.core.Dense object at 0x00000288ADB396A0>
<keras.layers.core.Dense object at 0x00000288ADB39A00>
<keras.layers.core.Dense object at 0x00000288ADB39D30>
tf.Tensor(
[[0.4033316  0.44451186 0.47364026 ... 0.48927888 0.5024994  0.58234787]
 [0.49599725 0.57751906 0.5915398  ... 0.48518807 0.34777355 0.6358223 ]
 [0.42722803 0.3749345  0.49382028 ... 0.48357564 0.48916495 0.5819093 ]
 ...
 [0.49081236 0.637867   0.25487703 ... 0.43735927 0.46753037 0.4927262 ]
 [0.4931479  0.32100934 0.63874567 ... 0.45376396 0.4281575  0.6749081 ]
 [0.33124268 0.52890104 0.40321895 ... 0.6630669  0.52765226 0.5764524 ]], shape=(32, 256), dtype=float32)
tf.Tensor(
[[[0.0950514 ]
  [0.09516188]
  [0.08048089]
  [0.08318727]
  [0.09785796]
  [0.08136254]
  [0.12293513]
  [0.1504391 ]
  [0.10127565]
  [0.09224822]]

 [[0.08785599]
  [0.08834638]
  [0.11902828]
  [0.07077389]
  [0.07329315]
  [0.08517081]
  [0.14497216]
  [0.09464727]
  [0.10208458]
  [0.13382743]]

 [[0.0574621 ]
  [0.06129359]
  [0.09928574]
  [0.14620145]
  [0.09131376]
  [0.08605367]
  [0.1211172 ]
  [0.11206973]
  [0.04775553]
  [0.1774472 ]]

 [[0.10692205]
  [0.0789185 ]
  [0.07715162]
  [0.09420718]
  [0.09866963]
  [0.077831  ]
  [0.11171441]
  [0.10380071]
  [0.14348388]
  [0.10730106]]

 [[0.08701313]
  [0.12229156]
  [0.08445423]
  [0.1100819 ]
  [0.09254183]
  [0.11487733]
  [0.06092561]
  [0.11954298]
  [0.08757055]
  [0.12070085]]

 [[0.1620955 ]
  [0.11929505]
  [0.09604067]
  [0.05728008]
  [0.10024205]
  [0.10547482]
  [0.07239203]
  [0.08347925]
  [0.10937135]
  [0.09432925]]

 [[0.11556066]
  [0.08797795]
  [0.08513419]
  [0.07566598]
  [0.15397182]
  [0.1064162 ]
  [0.07841221]
  [0.1144271 ]
  [0.10303292]
  [0.07940098]]

 [[0.1158981 ]
  [0.08636781]
  [0.09355568]
  [0.11081523]
  [0.10686832]
  [0.1281451 ]
  [0.10810142]
  [0.06974319]
  [0.08864968]
  [0.0918555 ]]

 [[0.08684821]
  [0.07456077]
  [0.1178768 ]
  [0.1292406 ]
  [0.08020458]
  [0.13921905]
  [0.09036247]
  [0.09590206]
  [0.08317744]
  [0.10260806]]

 [[0.1260529 ]
  [0.0738411 ]
  [0.0961778 ]
  [0.16015069]
  [0.09384325]
  [0.06872247]
  [0.0915657 ]
  [0.08261611]
  [0.10851724]
  [0.09851276]]

 [[0.09128509]
  [0.11598235]
  [0.10937162]
  [0.15307538]
  [0.06868506]
  [0.08781559]
  [0.11457431]
  [0.09477916]
  [0.07966731]
  [0.08476408]]

 [[0.1579263 ]
  [0.13137393]
  [0.06510878]
  [0.09789447]
  [0.07887884]
  [0.0879945 ]
  [0.06913688]
  [0.09976676]
  [0.07749307]
  [0.1344265 ]]

 [[0.07959273]
  [0.10207932]
  [0.05964907]
  [0.11586732]
  [0.11123122]
  [0.10498788]
  [0.08904688]
  [0.0905572 ]
  [0.11457638]
  [0.13241193]]

 [[0.16520281]
  [0.06923807]
  [0.0840843 ]
  [0.06432617]
  [0.06866088]
  [0.12871125]
  [0.12297405]
  [0.14073378]
  [0.07390904]
  [0.08215958]]

 [[0.06246531]
  [0.10400233]
  [0.11549831]
  [0.08112603]
  [0.07628184]
  [0.14527561]
  [0.14465533]
  [0.08745007]
  [0.0921874 ]
  [0.09105777]]

 [[0.07783865]
  [0.1190491 ]
  [0.11267986]
  [0.11038708]
  [0.10020015]
  [0.09066706]
  [0.07036457]
  [0.12863377]
  [0.08078108]
  [0.10939869]]

 [[0.0608872 ]
  [0.10971352]
  [0.10609878]
  [0.0937389 ]
  [0.09610511]
  [0.12611727]
  [0.09194995]
  [0.13553494]
  [0.08326121]
  [0.0965931 ]]

 [[0.08024353]
  [0.17055777]
  [0.07046402]
  [0.05193058]
  [0.08612729]
  [0.09465899]
  [0.13200562]
  [0.10792394]
  [0.09166358]
  [0.11442469]]

 [[0.08110558]
  [0.09680162]
  [0.11524588]
  [0.10330357]
  [0.09931505]
  [0.09187775]
  [0.10332955]
  [0.07442804]
  [0.11529317]
  [0.11929978]]

 [[0.09824485]
  [0.07409429]
  [0.09056538]
  [0.10485763]
  [0.10863946]
  [0.10659792]
  [0.09047807]
  [0.12226238]
  [0.11192154]
  [0.09233843]]

 [[0.11548259]
  [0.12635428]
  [0.07951286]
  [0.10928939]
  [0.08729077]
  [0.10919832]
  [0.11815677]
  [0.09140334]
  [0.08231254]
  [0.08099912]]

 [[0.10372857]
  [0.08917355]
  [0.12110065]
  [0.07837327]
  [0.0963261 ]
  [0.09630837]
  [0.09462981]
  [0.07231119]
  [0.14250857]
  [0.10553999]]

 [[0.06806559]
  [0.13353352]
  [0.08703799]
  [0.11249592]
  [0.1271834 ]
  [0.07957751]
  [0.08451825]
  [0.07138172]
  [0.13350363]
  [0.10270251]]

 [[0.06334788]
  [0.08314304]
  [0.08604384]
  [0.08132567]
  [0.10845412]
  [0.08327542]
  [0.12233253]
  [0.1090031 ]
  [0.13105999]
  [0.13201448]]

 [[0.07338195]
  [0.09779707]
  [0.19201745]
  [0.09388078]
  [0.11309531]
  [0.09487125]
  [0.08862841]
  [0.07832266]
  [0.05974443]
  [0.10826079]]

 [[0.10564204]
  [0.11753265]
  [0.07110972]
  [0.09310582]
  [0.14621404]
  [0.0450746 ]
  [0.09299395]
  [0.09180225]
  [0.11047955]
  [0.12604544]]

 [[0.07740418]
  [0.13076033]
  [0.11589289]
  [0.09489783]
  [0.10308607]
  [0.07850888]
  [0.10067194]
  [0.11183804]
  [0.09242405]
  [0.09451583]]

 [[0.08794035]
  [0.09135365]
  [0.09417484]
  [0.11737709]
  [0.11208227]
  [0.11279842]
  [0.11009596]
  [0.07756092]
  [0.1233606 ]
  [0.07325587]]

 [[0.11849148]
  [0.06442273]
  [0.11190039]
  [0.12532848]
  [0.10009912]
  [0.05168843]
  [0.07483967]
  [0.11695433]
  [0.07838306]
  [0.15789227]]

 [[0.12632973]
  [0.07214146]
  [0.11370175]
  [0.13783145]
  [0.07646144]
  [0.08051679]
  [0.11856095]
  [0.12303123]
  [0.06200291]
  [0.08942223]]

 [[0.11372808]
  [0.12381742]
  [0.08873468]
  [0.09195965]
  [0.06946086]
  [0.08089125]
  [0.18456832]
  [0.08360107]
  [0.10716835]
  [0.05607022]]

 [[0.12383202]
  [0.08982681]
  [0.08951265]
  [0.06625869]
  [0.07633531]
  [0.0867817 ]
  [0.10876328]
  [0.12534629]
  [0.11054894]
  [0.12279425]]], shape=(32, 10, 1), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 1-self_attention.py
 
0/4 pts
2. RNN Decoder
mandatory
Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation:

Class constructor def __init__(self, vocab, embedding, units, batch):
vocab is an integer representing the size of the output vocabulary
embedding is an integer representing the dimensionality of the embedding vector
units is an integer representing the number of hidden units in the RNN cell
batch is an integer representing the batch size
Sets the following public instance attributes:
embedding - a keras Embedding layer that converts words from the vocabulary into an embedding vector
gru - a keras GRU layer with units units
Should return both the full sequence of outputs as well as the last hidden state
Recurrent weights should be initialized with glorot_uniform
F - a Dense layer with vocab units
Public instance method def call(self, x, s_prev, hidden_states):
x is a tensor of shape (batch, 1) containing the previous word in the target sequence as an index of the target vocabulary
s_prev is a tensor of shape (batch, units) containing the previous decoder hidden state
hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
You should use SelfAttention = __import__('1-self_attention').SelfAttention
You should concatenate the context vector with x in that order
Returns: y, s
y is a tensor of shape (batch, vocab) containing the output word as a one hot vector in the target vocabulary
s is a tensor of shape (batch, units) containing the new decoder hidden state
$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

decoder = RNNDecoder(2048, 128, 256, 32)
print(decoder.embedding)
print(decoder.gru)
print(decoder.F)
x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
y, s = decoder(x, s_prev, hidden_states)
print(y)
print(s)
$ ./2-main.py
<keras.layers.embeddings.Embedding object at 0x00000185AB1AD7F0>
<keras.layers.recurrent_v2.GRU object at 0x00000185AB1ADAC0>
<keras.layers.core.Dense object at 0x00000185AB1ADA90>
tf.Tensor(
[[ 0.1106549   0.12441391 -0.10649128 ... -0.15290493 -0.00915979
   0.02811671]
 [ 0.12857069  0.14964633 -0.11658687 ... -0.17033781 -0.01144248
   0.00672281]
 [ 0.12285218  0.13817212 -0.11951214 ... -0.16218324 -0.01973578
   0.01638121]
 ...
 [ 0.11529345  0.14558475 -0.11789463 ... -0.1678974  -0.00715937
  -0.0013202 ]
 [ 0.1146404   0.14865148 -0.12343802 ... -0.1522213  -0.02677372
   0.0409069 ]
 [ 0.14619276  0.15417254 -0.13384987 ... -0.15048008 -0.03312323
   0.02073708]], shape=(32, 2048), dtype=float32)
tf.Tensor(
[[-0.02306041 -0.1545896   0.04115371 ... -0.04606991  0.1572462
   0.06915883]
 [-0.04795454 -0.1262584  -0.07350826 ...  0.01661924  0.10196586
   0.03525759]
 [ 0.01991665 -0.09092066 -0.05927104 ... -0.01334091  0.13552022
   0.00679498]
 ...
 [-0.02738137 -0.09386874 -0.02236932 ...  0.02396866  0.14071181
  -0.02056504]
 [-0.02928505 -0.08384971 -0.04409549 ...  0.01517994  0.09704634
   0.03551703]
 [-0.04720818 -0.14023307 -0.04800744 ...  0.00599389  0.12421984
  -0.02604038]], shape=(32, 256), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 2-rnn_decoder.py
 
0/4 pts
3. Positional Encoding
mandatory
Write the function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer:

max_seq_len is an integer representing the maximum sequence length
dm is the model depth
Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the positional encoding vectors
You should use import numpy as np
$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding

PE = positional_encoding(30, 512)
print(PE.shape)
print(PE)
$ ./4-main.py
(30, 512)
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.56375928e-01 -2.92138809e-01  7.91416314e-01 ...  9.99995791e-01
   2.79890525e-03  9.99996083e-01]
 [ 2.70905788e-01 -9.62605866e-01  9.53248145e-01 ...  9.99995473e-01
   2.90256812e-03  9.99995788e-01]
 [-6.63633884e-01 -7.48057530e-01  2.94705106e-01 ...  9.99995144e-01
   3.00623096e-03  9.99995481e-01]]
$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 4-positional_encoding.py
 
0/3 pts
4. Scaled Dot Product Attention
mandatory


Write the function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention:

Q is a tensor with its last two dimensions as (..., seq_len_q, dk) containing the query matrix
K is a tensor with its last two dimensions as (..., seq_len_v, dk) containing the key matrix
V is a tensor with its last two dimensions as (..., seq_len_v, dv) containing the value matrix
mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v) containing the optional mask, or defaulted to None
if mask is not None, multiply -1e9 to the mask and add it to the scaled matrix multiplication
The preceding dimensions of Q, K, and V are the same
Returns: output, weights
outputa tensor with its last two dimensions as (..., seq_len_q, dv) containing the scaled dot product attention
weights a tensor with its last two dimensions as (..., seq_len_q, seq_len_v) containing the attention weights
$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
$ ./5-main.py
tf.Tensor(
[[[0.5260249  0.6330511  0.39105493 ... 0.50381804 0.31397834 0.5435244 ]
  [0.5326587  0.6363233  0.38520008 ... 0.50928164 0.320748   0.529545  ]
  [0.5256397  0.64351934 0.39035168 ... 0.5124343  0.31304792 0.53604084]
  ...
  [0.52280194 0.6487294  0.3935972  ... 0.51527774 0.3181779  0.5334744 ]
  [0.523515   0.64202905 0.39964774 ... 0.5046137  0.31423992 0.53647566]
  [0.5354594  0.6390406  0.3939686  ... 0.51529115 0.33709148 0.5237254 ]]

 [[0.56073505 0.5006992  0.4356553  ... 0.56711316 0.6240922  0.48981497]
  [0.5641952  0.49763635 0.43044987 ... 0.5685781  0.61222744 0.48184517]
  [0.5606692  0.4981955  0.4321674  ... 0.5569761  0.6272035  0.48412377]
  ...
  [0.573033   0.5003185  0.42422023 ... 0.5688201  0.6166382  0.4944699 ]
  [0.5682367  0.49352264 0.4131574  ... 0.55753684 0.6130289  0.47994372]
  [0.5613553  0.48653772 0.42484856 ... 0.5619513  0.62081414 0.47832364]]

 [[0.70253754 0.60506254 0.43671212 ... 0.42608908 0.54443914 0.37897167]
  [0.6992751  0.5988886  0.4293988  ... 0.4323234  0.54449195 0.37536138]
  [0.7081644  0.6081345  0.42690748 ... 0.4270886  0.53528196 0.38357607]
  ...
  [0.69683355 0.6081301  0.42366233 ... 0.4378982  0.5407651  0.3780434 ]
  [0.70947653 0.6182213  0.42673317 ... 0.42354012 0.5468286  0.37565246]
  [0.7018702  0.6009647  0.42163122 ... 0.44308093 0.5451352  0.37584642]]

 ...

 [[0.5427401  0.5532219  0.51306427 ... 0.48199362 0.5580866  0.42930388]
  [0.56089985 0.54772353 0.50424147 ... 0.4826231  0.55346245 0.44748476]
  [0.54720336 0.5374128  0.5015749  ... 0.48345494 0.559634   0.44288322]
  ...
  [0.550317   0.5470272  0.4943997  ... 0.4914166  0.5439843  0.4463856 ]
  [0.5412615  0.5519809  0.5080425  ... 0.4797057  0.5538593  0.43696308]
  [0.545729   0.5689644  0.50514305 ... 0.48057818 0.5599982  0.43025455]]

 [[0.47995257 0.5100338  0.54621834 ... 0.5820797  0.54734856 0.5365638 ]
  [0.48361108 0.5099954  0.5301611  ... 0.5744918  0.55238765 0.5357528 ]
  [0.4787047  0.5128744  0.5350819  ... 0.5807635  0.55455893 0.5287215 ]
  ...
  [0.485692   0.5185557  0.52765113 ... 0.56333923 0.5374458  0.5381404 ]
  [0.4845507  0.50556517 0.53261834 ... 0.5719753  0.55349493 0.5272971 ]
  [0.48519212 0.5142199  0.544242   ... 0.5710808  0.54709464 0.5372726 ]]

 [[0.41277793 0.522753   0.40873203 ... 0.4702747  0.44156486 0.38912168]
  [0.4129986  0.5253192  0.4149088  ... 0.47701535 0.43706247 0.38952333]
  [0.41592228 0.5346544  0.41219917 ... 0.48161653 0.43787813 0.3925515 ]
  ...
  [0.4176696  0.5246175  0.40685236 ... 0.46955898 0.45170647 0.39105713]
  [0.4124788  0.52437747 0.4168522  ... 0.48022708 0.43835685 0.39118165]
  [0.39776435 0.52396995 0.39901358 ... 0.46348628 0.4465359  0.38358253]]], shape=(50, 10, 512), dtype=float32)
tf.Tensor(
[[[0.05747645 0.06372689 0.07961767 ... 0.06773182 0.06879826 0.0675021 ]
  [0.06385133 0.05926835 0.06914535 ... 0.07004026 0.07495442 0.06601514]
  [0.06615742 0.06291156 0.06941017 ... 0.07846092 0.08216266 0.06167598]
  ...
  [0.06432781 0.05953028 0.06566172 ... 0.06924983 0.07715577 0.07116533]
  [0.07097633 0.06562163 0.067275   ... 0.06190147 0.07136749 0.06474492]
  [0.0639073  0.06399637 0.06683359 ... 0.0894964  0.06345277 0.08068315]]

 [[0.06167933 0.06756181 0.06692738 ... 0.05943755 0.06113582 0.07553294]
  [0.06028167 0.06889632 0.07396711 ... 0.07046806 0.06308763 0.06348613]
  [0.07235043 0.06883418 0.07884171 ... 0.06225076 0.06303795 0.07420202]
  ...
  [0.05799915 0.06249382 0.07658647 ... 0.06157804 0.06288024 0.07477134]
  [0.0631256  0.06814098 0.08416418 ... 0.06636997 0.07142495 0.06012341]
  [0.07210932 0.07139985 0.06687157 ... 0.05975612 0.07890198 0.06475323]]

 [[0.07380512 0.06065308 0.07163459 ... 0.04355195 0.10245585 0.0715502 ]
  [0.07420148 0.05372792 0.06393997 ... 0.04826191 0.09912007 0.08020143]
  [0.08215743 0.06249476 0.07202116 ... 0.05402032 0.0925697  0.07716864]
  ...
  [0.07405666 0.0580003  0.05884973 ... 0.06822369 0.08682594 0.08559722]
  [0.07794132 0.05850713 0.06877114 ... 0.0564301  0.10531943 0.07123381]
  [0.07362355 0.04340661 0.06870168 ... 0.0593456  0.09257056 0.06778359]]

 ...

 [[0.06430291 0.07173725 0.06000953 ... 0.06867946 0.06663976 0.06857679]
  [0.06345265 0.07476585 0.0654313  ... 0.06826521 0.05566118 0.06462144]
  [0.06041237 0.07654066 0.07604242 ... 0.07577395 0.06419858 0.05583268]
  ...
  [0.05513643 0.07376103 0.0728935  ... 0.06959164 0.07767837 0.0663451 ]
  [0.06474954 0.06200786 0.0649054  ... 0.07864972 0.0717388  0.06216607]
  [0.06139709 0.06485597 0.0670691  ... 0.05706039 0.06033475 0.07405537]]

 [[0.05736551 0.0889615  0.07514869 ... 0.06784085 0.07893378 0.04980288]
  [0.06226587 0.0928518  0.06383356 ... 0.0627     0.07146592 0.05461694]
  [0.05954041 0.09551187 0.05904439 ... 0.06896569 0.07033826 0.05058166]
  ...
  [0.06606079 0.09496784 0.06873915 ... 0.05659967 0.06069246 0.05936281]
  [0.0515007  0.09925249 0.05824476 ... 0.06311911 0.06792571 0.04967246]
  [0.0604107  0.10163556 0.06725224 ... 0.07251322 0.07478119 0.04964384]]

 [[0.05551238 0.07867253 0.0650571  ... 0.08090825 0.0566463  0.05778079]
  [0.05219174 0.06973514 0.06533503 ... 0.078656   0.06487647 0.05779628]
  [0.05759448 0.06036389 0.07039176 ... 0.07936198 0.06222914 0.04924813]
  ...
  [0.05004732 0.07064635 0.06901565 ... 0.07795537 0.05310345 0.06313733]
  [0.059706   0.06734002 0.06407155 ... 0.07299846 0.05878944 0.05612334]
  [0.05442357 0.07557689 0.06472676 ... 0.06223008 0.06921906 0.06195656]]], shape=(50, 10, 15), dtype=float32)
$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 5-sdp_attention.py
 
0/3 pts
5. Multi Head Attention
mandatory


Read:

Why multi-head self attention works: math, intuitions and 10+1 hidden insights
Create a class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention:

Class constructor def __init__(self, dm, h):
dm is an integer representing the dimensionality of the model
h is an integer representing the number of heads
dm is divisible by h
Sets the following public instance attributes:
h - the number of heads
dm - the dimensionality of the model
depth - the depth of each attention head
Wq - a Dense layer with dm units, used to generate the query matrix
Wk - a Dense layer with dm units, used to generate the key matrix
Wv - a Dense layer with dm units, used to generate the value matrix
linear - a Dense layer with dm units, used to generate the attention output
Public instance method def call(self, Q, K, V, mask):
Q is a tensor of shape (batch, seq_len_q, dk) containing the input to generate the query matrix
K is a tensor of shape (batch, seq_len_v, dk) containing the input to generate the key matrix
V is a tensor of shape (batch, seq_len_v, dv) containing the input to generate the value matrix
mask is always None
Returns: output, weights
outputa tensor with its last two dimensions as (..., seq_len_q, dm) containing the scaled dot product attention
weights a tensor with its last three dimensions as (..., h, seq_len_q, seq_len_v) containing the attention weights
You should use sdp_attention = __import__('5-sdp_attention').sdp_attention
$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

mha = MultiHeadAttention(512, 8)
print(mha.dm)
print(mha.h)
print(mha.depth)
print(mha.Wq)
print(mha.Wk)
print(mha.Wv)
print(mha.linear)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
output, weights = mha(Q, K, V, None)
print(output)
print(weights)
$ ./6-main.py
512
8
64
<keras.layers.core.Dense object at 0x00000129A442B760>
<keras.layers.core.Dense object at 0x00000129A442BBE0>
<keras.layers.core.Dense object at 0x00000129A442BEE0>
<keras.layers.core.Dense object at 0x00000129A44B3220>
tf.Tensor(
[[[-0.34425578 -0.18614006  0.24008222 ...  0.34599832 -0.2567526
    0.45626754]
  [-0.3402937  -0.19175807  0.2379572  ...  0.33900094 -0.25409284
    0.4479677 ]
  [-0.3520621  -0.188232    0.23780414 ...  0.3424595  -0.25620028
    0.45275515]
  ...
  [-0.34192288 -0.188041    0.23590818 ...  0.34419012 -0.25593722
    0.45327145]
  [-0.34883115 -0.18573773  0.23332804 ...  0.33914775 -0.25613534
    0.4483503 ]
  [-0.34207255 -0.18439627  0.23783532 ...  0.34494442 -0.24964732
    0.4563626 ]]

 [[-0.3058339  -0.21900034  0.2996024  ...  0.358339   -0.33864132
    0.4889792 ]
  [-0.29965663 -0.22172472  0.29042393 ...  0.35751742 -0.3379998
    0.4750886 ]
  [-0.30475146 -0.22462834  0.29387808 ...  0.35763553 -0.33610585
    0.4779113 ]
  ...
  [-0.30292854 -0.21851459  0.30133215 ...  0.3598636  -0.33952206
    0.48519623]
  [-0.3017973  -0.22162111  0.29391596 ...  0.35353896 -0.33895087
    0.48077363]
  [-0.298313   -0.22350118  0.29639935 ...  0.3514393  -0.33894736
    0.47818044]]

 [[-0.34826538 -0.19922528  0.288432   ...  0.44866985 -0.43255478
    0.5437051 ]
  [-0.35370988 -0.20121038  0.29943952 ...  0.44904634 -0.4280138
    0.54305613]
  [-0.34932598 -0.21157122  0.29594678 ...  0.4489977  -0.43952873
    0.5397042 ]
  ...
  [-0.34940556 -0.2024809   0.29071504 ...  0.4534883  -0.43239257
    0.5378097 ]
  [-0.3513743  -0.19665268  0.2966078  ...  0.45449644 -0.42896485
    0.54231334]
  [-0.3481236  -0.20059955  0.29704392 ...  0.45813712 -0.423687
    0.5452433 ]]

 ...

 [[-0.2786278  -0.03083764  0.21793929 ...  0.53628707 -0.35585773
    0.47551268]
  [-0.2791264  -0.03123678  0.22002706 ...  0.53772867 -0.34568164
    0.4645136 ]
  [-0.2684938  -0.03472327  0.21968988 ...  0.5383915  -0.34750396
    0.46525276]
  ...
  [-0.26842564 -0.03112899  0.21575662 ...  0.5425405  -0.34350282
    0.46796137]
  [-0.27474064 -0.02786241  0.21464701 ...  0.5382316  -0.3452933
    0.47530314]
  [-0.2803016  -0.03340159  0.2174516  ...  0.5352315  -0.34866625
    0.47715983]]

 [[-0.3951821  -0.18505685  0.1894601  ...  0.39745727 -0.171678
    0.4017078 ]
  [-0.39544043 -0.18047109  0.18747053 ...  0.392758   -0.17420724
    0.40179414]
  [-0.3900192  -0.18497609  0.18076494 ...  0.39703035 -0.17885894
    0.38981307]
  ...
  [-0.39612457 -0.18725036  0.1831272  ...  0.40168762 -0.17330864
    0.3975389 ]
  [-0.39667132 -0.1825059   0.18523388 ...  0.39519382 -0.16879499
    0.40367743]
  [-0.39891368 -0.18755378  0.18579349 ...  0.39689755 -0.17671959
    0.40131938]]

 [[-0.3322693  -0.15948185  0.29808924 ...  0.39094937 -0.24646084
    0.48524386]
  [-0.33442026 -0.1644111   0.3034696  ...  0.394661   -0.25570765
    0.4881563 ]
  [-0.33862334 -0.15872703  0.30237684 ...  0.392514   -0.2451564
    0.48560396]
  ...
  [-0.33562884 -0.15932046  0.3029383  ...  0.39129835 -0.24836294
    0.48867416]
  [-0.3321737  -0.16020423  0.300864   ...  0.3961613  -0.25021684
    0.48327947]
  [-0.3374202  -0.15894113  0.29939485 ...  0.39029837 -0.2501283
    0.48372114]]], shape=(50, 15, 512), dtype=float32)
tf.Tensor(
[[[[0.06314778 0.05786009 0.07427105 ... 0.06843876 0.08241873
    0.06699952]
   [0.06600787 0.06277192 0.07166055 ... 0.06603511 0.07838564
    0.06577787]
   [0.06225236 0.05747318 0.07562194 ... 0.06647826 0.083197
    0.06050787]
   ...
   [0.06831431 0.05599761 0.07199641 ... 0.06963189 0.08134753
    0.06019705]
   [0.06449199 0.05869886 0.06686782 ... 0.07449434 0.08068755
    0.06208705]
   [0.06354798 0.06060821 0.07305495 ... 0.06114892 0.07700062
    0.06361192]]

  [[0.05423188 0.06123925 0.06302712 ... 0.06023842 0.07381344
    0.06787337]
   [0.05403511 0.06595989 0.06620718 ... 0.06365778 0.06455199
    0.07041122]
   [0.05881567 0.06248171 0.07176445 ... 0.06723251 0.06464646
    0.06129514]
   ...
   [0.06689195 0.06572177 0.07052002 ... 0.05932291 0.06214605
    0.06788114]
   [0.06535013 0.05810506 0.06445628 ... 0.06801771 0.06094166
    0.06685688]
   [0.05506894 0.05731435 0.06027539 ... 0.0637766  0.07032777
    0.07035653]]

  [[0.06190565 0.07369512 0.05850432 ... 0.0630995  0.07688209
    0.06225207]
   [0.07254362 0.07241444 0.05218989 ... 0.05921613 0.07339299
    0.0623728 ]
   [0.06818843 0.06579394 0.06180909 ... 0.06347412 0.06519561
    0.06474791]
   ...
   [0.06996431 0.07877766 0.05888139 ... 0.06262539 0.06531936
    0.06592741]
   [0.06573584 0.07702287 0.05074254 ... 0.06424327 0.06642514
    0.06463128]
   [0.06558772 0.07159246 0.05878147 ... 0.06255294 0.07152926
    0.06440628]]

  ...

  [[0.06749792 0.05998375 0.06592966 ... 0.06940931 0.07535786
    0.06619785]
   [0.07508499 0.06656995 0.06332656 ... 0.0697884  0.06872913
    0.06944171]
   [0.06533524 0.0643831  0.06146031 ... 0.07795397 0.0701353
    0.06716793]
   ...
   [0.06489884 0.05739467 0.05995832 ... 0.07595459 0.07065734
    0.06870002]
   [0.06657062 0.06548417 0.06801336 ... 0.06783177 0.07541881
    0.06394814]
   [0.07140248 0.05640087 0.06382447 ... 0.07456069 0.06985733
    0.06788827]]

  [[0.07080363 0.06019336 0.06193018 ... 0.08014785 0.06897142
    0.06596588]
   [0.07214461 0.06474414 0.06325515 ... 0.07409827 0.06440809
    0.07668527]
   [0.06784151 0.06000459 0.07626323 ... 0.08249928 0.06888764
    0.06853784]
   ...
   [0.06521757 0.05510813 0.06993075 ... 0.08220559 0.06571949
    0.06459403]
   [0.06484979 0.05504489 0.07157404 ... 0.07561293 0.06475756
    0.07268962]
   [0.07002608 0.05945696 0.06821597 ... 0.07688458 0.06266746
    0.0661274 ]]

  [[0.06395745 0.06768958 0.06107244 ... 0.06063186 0.06830138
    0.0963348 ]
   [0.06172961 0.06835959 0.05647578 ... 0.05810838 0.06827948
    0.0857655 ]
   [0.06290781 0.0652248  0.06327426 ... 0.06145196 0.07279003
    0.09903048]
   ...
   [0.06148448 0.0714151  0.06010536 ... 0.06760606 0.06498186
    0.08537357]
   [0.06153386 0.06071218 0.05745455 ... 0.0623216  0.0637159
    0.08921374]
   [0.0647856  0.07043882 0.06593192 ... 0.05993811 0.0697723
    0.08835946]]]


 [[[0.07133106 0.06376223 0.07462363 ... 0.05452179 0.05884367
    0.0676857 ]
   [0.06756676 0.06794378 0.0656867  ... 0.06050397 0.05689743
    0.06254524]
   [0.06695356 0.06756931 0.06450099 ... 0.05595957 0.05927519
    0.06415462]
   ...
   [0.06579109 0.0638969  0.06878292 ... 0.05453823 0.07049747
    0.06233548]
   [0.06107162 0.0636906  0.0771208  ... 0.06018046 0.05023453
    0.07139777]
   [0.072579   0.06057664 0.06674537 ... 0.05970381 0.05857566
    0.06244272]]

  [[0.0657206  0.06623848 0.06408031 ... 0.06146023 0.0720204
    0.07387258]
   [0.06452714 0.06615207 0.05938598 ... 0.0600872  0.07105564
    0.08016796]
   [0.06441949 0.06769315 0.0643999  ... 0.06267282 0.07800475
    0.07629834]
   ...
   [0.06574256 0.066705   0.06547249 ... 0.06131643 0.08315425
    0.07684937]
   [0.06588022 0.06587993 0.06109166 ... 0.06464684 0.07506178
    0.08808874]
   [0.06238504 0.05995611 0.05954672 ... 0.05897006 0.07889587
    0.08423989]]

  [[0.06506601 0.06304572 0.07506898 ... 0.07970189 0.05430445
    0.067041  ]
   [0.06513147 0.06645182 0.06779321 ... 0.08627807 0.0712591
    0.07397455]
   [0.0628427  0.06681503 0.0719998  ... 0.07898752 0.06213209
    0.062819  ]
   ...
   [0.0700554  0.06515904 0.07640522 ... 0.08197132 0.06110426
    0.06572595]
   [0.07146494 0.07192253 0.06604601 ... 0.08052576 0.06089837
    0.06925917]
   [0.07174478 0.06673909 0.06935875 ... 0.07292478 0.06054124
    0.06157237]]

  ...

  [[0.07450838 0.06281502 0.06880775 ... 0.0664423  0.06220401
    0.07538728]
   [0.06918153 0.05618425 0.06552384 ... 0.07158619 0.06387391
    0.06253017]
   [0.06793844 0.06165997 0.06868319 ... 0.06920971 0.06616934
    0.070062  ]
   ...
   [0.07034242 0.05939212 0.07097111 ... 0.06813059 0.06043071
    0.07188566]
   [0.07146574 0.06267934 0.06322124 ... 0.0675593  0.05950634
    0.06928383]
   [0.07012586 0.06097022 0.07087758 ... 0.07464605 0.06822183
    0.06603421]]

  [[0.07784256 0.0501098  0.06811247 ... 0.06261865 0.0667598
    0.06791607]
   [0.08199309 0.0581786  0.06406369 ... 0.06825669 0.0547633
    0.07507231]
   [0.07303347 0.0560666  0.07471286 ... 0.0572117  0.05945962
    0.07295388]
   ...
   [0.07465532 0.05365309 0.06942465 ... 0.0620176  0.06530015
    0.06505259]
   [0.07827984 0.05390388 0.06543186 ... 0.06398238 0.06104857
    0.06930588]
   [0.07068281 0.06160644 0.0646641  ... 0.06520949 0.07069157
    0.06304686]]

  [[0.06686892 0.05881336 0.05968849 ... 0.07648427 0.0573192
    0.07553767]
   [0.06498811 0.06873461 0.06586123 ... 0.07747953 0.05802261
    0.07087497]
   [0.06687935 0.0628935  0.06523469 ... 0.07969883 0.05946669
    0.0751478 ]
   ...
   [0.06232864 0.0606367  0.06720933 ... 0.07589602 0.06545808
    0.07338448]
   [0.07259546 0.07122415 0.06144801 ... 0.07270242 0.05839765
    0.06594798]
   [0.05844144 0.06070074 0.06854649 ... 0.07888718 0.06333768
    0.06912053]]]


 [[[0.07077207 0.08817283 0.05793432 ... 0.06842075 0.06478342
    0.05902337]
   [0.06886206 0.07535799 0.06518483 ... 0.06567556 0.06852701
    0.05997531]
   [0.06972644 0.07749464 0.0599536  ... 0.07087991 0.06834663
    0.06202123]
   ...
   [0.0705298  0.08175028 0.05874606 ... 0.06832212 0.06603503
    0.05876396]
   [0.06655259 0.08446684 0.06347395 ... 0.07165792 0.06526842
    0.06208567]
   [0.07756389 0.07298542 0.05797721 ... 0.06434692 0.06751989
    0.06125375]]

  [[0.06581312 0.06514276 0.07989842 ... 0.06641971 0.07312226
    0.06998309]
   [0.06096456 0.06879506 0.0862491  ... 0.06928728 0.06574637
    0.06488297]
   [0.06254108 0.06879535 0.08195303 ... 0.06754886 0.07014713
    0.07329457]
   ...
   [0.0602859  0.06856477 0.08236988 ... 0.07139678 0.07059322
    0.06562411]
   [0.05897532 0.06651524 0.08561656 ... 0.06698922 0.0655914
    0.06974186]
   [0.05875244 0.06566602 0.08405764 ... 0.06733575 0.06966954
    0.07451425]]

  [[0.06703417 0.06728691 0.06289449 ... 0.06358974 0.07448665
    0.07560294]
   [0.06091861 0.06983574 0.07073572 ... 0.05718503 0.07670788
    0.07580012]
   [0.06186581 0.08043326 0.06580627 ... 0.06216057 0.07277294
    0.07637487]
   ...
   [0.06573984 0.07355092 0.07082853 ... 0.0629857  0.07203959
    0.07492593]
   [0.06088949 0.063697   0.07666945 ... 0.05869371 0.07854827
    0.08029764]
   [0.06187497 0.06384401 0.07108877 ... 0.06088396 0.0765968
    0.08162133]]

  ...

  [[0.05874245 0.07729736 0.06296913 ... 0.06969842 0.07311659
    0.06022416]
   [0.05954337 0.07597046 0.0563589  ... 0.06515809 0.06880026
    0.06504297]
   [0.06259734 0.07682964 0.05489504 ... 0.06084017 0.06813752
    0.05962181]
   ...
   [0.06392267 0.07661688 0.05892519 ... 0.06469644 0.06665064
    0.05514186]
   [0.06164048 0.07630214 0.0584657  ... 0.06916331 0.07088767
    0.06089139]
   [0.06179587 0.07466681 0.06269313 ... 0.07085993 0.07025144
    0.06566481]]

  [[0.06252273 0.06742212 0.07186425 ... 0.07365113 0.06173573
    0.05944533]
   [0.06473549 0.05778869 0.06839802 ... 0.06862868 0.06337976
    0.05876067]
   [0.07235486 0.06142993 0.07156432 ... 0.06774683 0.05992295
    0.06666613]
   ...
   [0.06522964 0.06441628 0.07269571 ... 0.06891205 0.05894208
    0.06646062]
   [0.06610013 0.05849081 0.0754272  ... 0.06725474 0.06010116
    0.06278956]
   [0.06836442 0.0613678  0.07503507 ... 0.07186269 0.06227709
    0.06341568]]

  [[0.06468728 0.06892566 0.06903619 ... 0.05875579 0.06078074
    0.05750612]
   [0.0650343  0.0679981  0.07084385 ... 0.0585239  0.05952591
    0.05563875]
   [0.06621559 0.06948256 0.06972967 ... 0.06339408 0.0607331
    0.05558217]
   ...
   [0.06248083 0.06581984 0.07310209 ... 0.06073539 0.05974793
    0.06251112]
   [0.06204985 0.0700525  0.06935238 ... 0.0622101  0.05824918
    0.06369138]
   [0.06836464 0.07101926 0.06465451 ... 0.06135549 0.05752216
    0.06335159]]]


 ...


 [[[0.06187404 0.07242148 0.07425116 ... 0.06700491 0.06514528
    0.06026656]
   [0.0599307  0.06889077 0.07405475 ... 0.06642727 0.054799
    0.06659496]
   [0.05789003 0.0636133  0.07046193 ... 0.06761955 0.05910896
    0.07106184]
   ...
   [0.06079898 0.06953687 0.07187951 ... 0.0622322  0.05811676
    0.0674249 ]
   [0.06751208 0.06865422 0.06671734 ... 0.06180595 0.06326323
    0.07044435]
   [0.06071197 0.06990835 0.06657916 ... 0.06458278 0.06049525
    0.07768334]]

  [[0.07688951 0.06666427 0.06517886 ... 0.0599755  0.05925946
    0.07866984]
   [0.07487942 0.06542087 0.06504336 ... 0.0640736  0.07054418
    0.07966644]
   [0.07559343 0.06881065 0.0625417  ... 0.06224044 0.07071109
    0.08540376]
   ...
   [0.06998616 0.07214639 0.06073333 ... 0.06305249 0.07071912
    0.08293968]
   [0.07563349 0.06585933 0.06680302 ... 0.06289995 0.06565449
    0.08192916]
   [0.07230607 0.05927031 0.06994559 ... 0.0703634  0.06240659
    0.07756323]]

  [[0.05448047 0.06374928 0.06661247 ... 0.07426748 0.0679069
    0.05421144]
   [0.06330046 0.06757879 0.06504134 ... 0.07795547 0.06638727
    0.05754854]
   [0.06334188 0.0788899  0.07088366 ... 0.06778944 0.06606897
    0.05619345]
   ...
   [0.05886443 0.06958837 0.06934011 ... 0.06865536 0.06853545
    0.05521048]
   [0.05949087 0.06841081 0.07162318 ... 0.07034867 0.06562708
    0.05489   ]
   [0.05959066 0.0705574  0.07433308 ... 0.07340097 0.05996491
    0.05070869]]

  ...

  [[0.06584992 0.05555017 0.06759135 ... 0.06881374 0.07413428
    0.06450675]
   [0.058118   0.05911113 0.06786394 ... 0.06811323 0.0739946
    0.06476324]
   [0.07068691 0.05282836 0.07660717 ... 0.07147791 0.07314334
    0.06791233]
   ...
   [0.06322068 0.06372975 0.0759894  ... 0.06351473 0.07042985
    0.06455428]
   [0.06023511 0.05458957 0.07166867 ... 0.06230006 0.07557172
    0.06741496]
   [0.06481507 0.05817467 0.0733906  ... 0.06563158 0.07421263
    0.06545469]]

  [[0.08528691 0.07026435 0.07495845 ... 0.06371026 0.05498207
    0.06278221]
   [0.07830912 0.06688236 0.07002079 ... 0.0682402  0.05680069
    0.05603665]
   [0.08635066 0.06363994 0.07137629 ... 0.06464013 0.05703694
    0.06125485]
   ...
   [0.06822651 0.06464415 0.06669168 ... 0.07064357 0.06021738
    0.06637104]
   [0.07542939 0.06336804 0.0696978  ... 0.06818166 0.06327331
    0.06629176]
   [0.08189983 0.07185407 0.06975155 ... 0.06641342 0.05986629
    0.05816509]]

  [[0.06994364 0.06043029 0.06661136 ... 0.06437121 0.07194167
    0.06314252]
   [0.06709832 0.06074093 0.06601959 ... 0.06378964 0.0653087
    0.05522685]
   [0.06983285 0.06246053 0.06723626 ... 0.06233827 0.06423523
    0.06630385]
   ...
   [0.07333902 0.05888303 0.06951174 ... 0.05964654 0.06741165
    0.06171881]
   [0.06813201 0.05763144 0.06462021 ... 0.07013968 0.06936473
    0.05835297]
   [0.06915462 0.06239427 0.06511137 ... 0.06149583 0.06328783
    0.05822326]]]


 [[[0.06515275 0.07421266 0.07076993 ... 0.07148774 0.06987945
    0.06218968]
   [0.06708691 0.06952167 0.06480067 ... 0.07756452 0.06299119
    0.06262689]
   [0.07040411 0.06758818 0.06535056 ... 0.06821039 0.06995412
    0.06777985]
   ...
   [0.06591266 0.06941493 0.06630295 ... 0.06934221 0.0684292
    0.06490432]
   [0.07129759 0.07334134 0.0631506  ... 0.06751476 0.05632473
    0.06487528]
   [0.07226496 0.06923059 0.06386444 ... 0.06872167 0.06266816
    0.06290194]]

  [[0.06168521 0.06155062 0.06112638 ... 0.06664927 0.07236842
    0.07339442]
   [0.06598402 0.06647974 0.05466088 ... 0.0670591  0.06595992
    0.07452913]
   [0.06298176 0.06233413 0.05476711 ... 0.06219298 0.07373875
    0.07561513]
   ...
   [0.0622295  0.05727932 0.05080051 ... 0.06926853 0.07402432
    0.08562773]
   [0.0628555  0.06240721 0.05851254 ... 0.06615172 0.06853323
    0.07694969]
   [0.06545691 0.06183948 0.05661534 ... 0.06193361 0.07405271
    0.07729967]]

  [[0.05701163 0.06669772 0.05608967 ... 0.07235055 0.05959526
    0.06120901]
   [0.06245551 0.06827547 0.05374447 ... 0.07016712 0.06129455
    0.06781484]
   [0.06437742 0.07381631 0.05471015 ... 0.06978341 0.05875938
    0.07370348]
   ...
   [0.06472742 0.07375389 0.04774416 ... 0.0711173  0.06620529
    0.06760977]
   [0.0606183  0.07709563 0.05499453 ... 0.07527604 0.06044722
    0.06581824]
   [0.0557945  0.08075652 0.05715513 ... 0.06840862 0.06341571
    0.07030528]]

  ...

  [[0.06274644 0.05846811 0.05201308 ... 0.07899234 0.0701282
    0.06052452]
   [0.06758279 0.06221201 0.06100909 ... 0.06873398 0.07131402
    0.05649845]
   [0.06885281 0.05773665 0.06123892 ... 0.06925796 0.07521171
    0.05587208]
   ...
   [0.06372928 0.0552677  0.05895336 ... 0.06920547 0.07110746
    0.05619961]
   [0.07245982 0.05103194 0.06216486 ... 0.07049219 0.07608426
    0.05307629]
   [0.06963602 0.05886824 0.05893583 ... 0.07003761 0.06952113
    0.06057265]]

  [[0.05745273 0.06418718 0.07649545 ... 0.06833104 0.06396139
    0.07021463]
   [0.05962847 0.06133353 0.07933889 ... 0.0702469  0.05925418
    0.0754159 ]
   [0.05714039 0.05947671 0.06920531 ... 0.07279418 0.0596465
    0.08156127]
   ...
   [0.07110482 0.05909343 0.05895212 ... 0.06504086 0.06421418
    0.06664317]
   [0.06037918 0.05832544 0.07077258 ... 0.06293183 0.05768512
    0.08056214]
   [0.06712695 0.05926772 0.06255215 ... 0.06794961 0.0594607
    0.07371565]]

  [[0.06226642 0.06502656 0.06781647 ... 0.06619569 0.06229878
    0.07259135]
   [0.06298137 0.0694892  0.08148452 ... 0.07066768 0.06147237
    0.06955383]
   [0.05928315 0.07794998 0.06751468 ... 0.07853765 0.06068844
    0.07233793]
   ...
   [0.06738363 0.06767835 0.07597327 ... 0.07118694 0.06236378
    0.07059662]
   [0.05751748 0.06741595 0.07445458 ... 0.07218243 0.0567386
    0.0691053 ]
   [0.06162539 0.07249935 0.07374073 ... 0.06911322 0.05505744
    0.07234159]]]


 [[[0.06080713 0.06702907 0.06584277 ... 0.05324364 0.08363958
    0.06168078]
   [0.06043154 0.05736104 0.06933025 ... 0.0632955  0.08094876
    0.05866491]
   [0.05658833 0.0726244  0.06483888 ... 0.06400322 0.08210078
    0.06031982]
   ...
   [0.05862856 0.06197324 0.07410521 ... 0.05948454 0.08215605
    0.05896702]
   [0.06263062 0.06198073 0.0706825  ... 0.05667934 0.08054707
    0.06558036]
   [0.05936761 0.0632505  0.06867424 ... 0.06410885 0.08026268
    0.05628729]]

  [[0.07144582 0.07123459 0.0593045  ... 0.06879992 0.07042427
    0.04788407]
   [0.07145837 0.06951674 0.06284316 ... 0.07121072 0.06830421
    0.05366172]
   [0.07566439 0.0635471  0.06214524 ... 0.06812946 0.07117864
    0.05515871]
   ...
   [0.07091258 0.06967387 0.06211216 ... 0.07037521 0.06938066
    0.05422853]
   [0.06753066 0.06793126 0.05791434 ... 0.06842311 0.06868615
    0.0567022 ]
   [0.0719948  0.07121129 0.06029432 ... 0.06883869 0.06734101
    0.05044019]]

  [[0.06738969 0.0780658  0.07236132 ... 0.06682463 0.06411114
    0.05573744]
   [0.06320318 0.07122546 0.06803964 ... 0.06841791 0.05918689
    0.06753536]
   [0.05820514 0.0703962  0.07226633 ... 0.06782568 0.06502988
    0.06158779]
   ...
   [0.06632905 0.07598684 0.07481771 ... 0.06250551 0.06078722
    0.06010192]
   [0.06531473 0.07127169 0.06989568 ... 0.06722955 0.05958243
    0.06495355]
   [0.06222012 0.07099239 0.06801874 ... 0.06601316 0.05299538
    0.06470978]]

  ...

  [[0.06143417 0.06722569 0.06254662 ... 0.07030601 0.06517965
    0.06695539]
   [0.07159082 0.068765   0.06751768 ... 0.07147225 0.05550396
    0.07386328]
   [0.06595692 0.06991737 0.06379723 ... 0.06879604 0.0606557
    0.06705328]
   ...
   [0.06645122 0.06403766 0.06913899 ... 0.07891772 0.0613571
    0.07398707]
   [0.06998061 0.07035766 0.06869663 ... 0.07822902 0.06671932
    0.06516666]
   [0.06471199 0.06132922 0.06290204 ... 0.07778174 0.0572699
    0.07606222]]

  [[0.06445111 0.05526718 0.06334258 ... 0.06305614 0.07546753
    0.06278694]
   [0.06276246 0.06134414 0.06177333 ... 0.05928209 0.07546845
    0.06698313]
   [0.07323315 0.05529476 0.069472   ... 0.06500117 0.07240625
    0.06580448]
   ...
   [0.06374808 0.05478458 0.06117201 ... 0.06844212 0.07467262
    0.07293735]
   [0.06407861 0.05485751 0.06847217 ... 0.06000594 0.06833561
    0.06797663]
   [0.06978642 0.05842579 0.07110637 ... 0.06258525 0.0748942
    0.0657613 ]]

  [[0.07302743 0.065016   0.06289599 ... 0.0596938  0.05089658
    0.06692782]
   [0.06878541 0.06783357 0.07383577 ... 0.05939251 0.05247147
    0.07225603]
   [0.0725619  0.06340981 0.06078735 ... 0.05835668 0.05698569
    0.07327002]
   ...
   [0.07276352 0.05698376 0.06383878 ... 0.05920573 0.05902321
    0.0672344 ]
   [0.07003184 0.06509699 0.06923351 ... 0.06230944 0.05066744
    0.0749735 ]
   [0.07090436 0.06199337 0.06152395 ... 0.06066424 0.0495515
    0.06994723]]]], shape=(50, 8, 15, 15), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 6-multihead_attention.py
 
0/4 pts
6. Transformer Encoder Block
mandatory


Create a class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer:

Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
dm - the dimensionality of the model
h - the number of heads
hidden - the number of hidden units in the fully connected layer
drop_rate - the dropout rate
Sets the following public instance attributes:
mha - a MultiHeadAttention layer
dense_hidden - the hidden dense layer with hidden units and relu activation
dense_output - the output dense layer with dm units
layernorm1 - the first layer norm layer, with epsilon=1e-6
layernorm2 - the second layer norm layer, with epsilon=1e-6
dropout1 - the first dropout layer
dropout2 - the second dropout layer
Public instance method call(self, x, training, mask=None):
x - a tensor of shape (batch, input_seq_len, dm)containing the input to the encoder block
training - a boolean to determine if the model is training
mask - the mask to be applied for multi head attention
Returns: a tensor of shape (batch, input_seq_len, dm) containing the block’s output
You should use MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention
$ cat 7-main
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

eblock = EncoderBlock(512, 8, 2048)
print(eblock.mha)
print(eblock.dense_hidden)
print(eblock.dense_output)
print(eblock.layernorm1)
print(eblock.layernorm2)
print(eblock.dropout1)
print(eblock.dropout2)
x = tf.random.uniform((32, 10, 512))
output = eblock(x, True, None)
print(output)
$ ./7-main.py
<6-multihead_attention.MultiHeadAttention object at 0x0000015D264FAA60>
<keras.layers.core.Dense object at 0x0000015D265847C0>
<keras.layers.core.Dense object at 0x0000015D26584C70>
<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000015D26584FA0>
<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000015D26595070>
<keras.layers.core.Dropout object at 0x0000015D265952E0>
<keras.layers.core.Dropout object at 0x0000015D26595550>
tf.Tensor(
[[[ 0.21383508 -0.31940326 -1.3922207  ... -0.89140785 -0.5559802
    0.00821821]
  [-0.6600582   0.3874671   0.10559016 ... -1.4396642  -0.76180124
    0.02207807]
  [-0.10975407  0.07995845 -1.5038341  ... -1.3607594   0.04429432
   -0.03273529]
  ...
  [ 0.00432308 -0.5270435  -1.7524896  ... -1.0212561   0.6079124
   -0.62109756]
  [-0.14458886 -0.5279723  -1.4416505  ... -0.83458406 -0.2861029
   -0.5691369 ]
  [-1.3577837  -0.382116   -1.7397237  ... -0.88560575 -0.88310564
   -0.2456336 ]]

 [[ 0.30616006 -0.58578223 -0.6243632  ... -0.93565    -0.20998262
   -1.0310975 ]
  [-0.05208162  0.31894973 -0.2859358  ... -1.9336534  -0.28001186
   -0.5454689 ]
  [ 0.34557185 -0.43803805 -0.2750871  ... -1.7654411  -0.4693619
    0.13281251]
  ...
  [ 0.12404484 -0.22363631 -0.21095781 ... -2.2666469  -0.4159829
   -0.13403216]
  [ 0.9969142   0.44125256 -1.3266566  ... -1.5723763  -0.20524824
   -0.33581114]
  [ 0.25062287 -0.15750915 -0.0291395  ... -0.98286843 -0.22284478
   -0.61358744]]

 [[-1.1452503   0.07249875 -1.3492672  ... -1.193882   -0.804781
   -0.4551219 ]
  [-0.491923   -0.43035805 -1.4572073  ... -0.46308938 -0.572636
   -0.20042765]
  [ 0.13628986 -0.8570299  -0.46575224 ... -1.2230774  -0.51441765
   -0.57393694]
  ...
  [-0.3213793  -1.3386041  -0.43547916 ... -0.78642476 -0.9553704
   -0.7802765 ]
  [-1.1408268  -0.40072787 -0.10889349 ... -1.2807635   0.36388674
    0.6244068 ]
  [ 0.08141474 -0.10364113 -1.4510003  ... -1.3006443   0.03428427
   -0.07453457]]

 ...

 [[-0.7369644   0.3885487  -2.6570923  ... -2.0945215  -1.1045542
   -0.06108083]
  [ 0.22220318 -1.0532161   0.60011894 ... -0.8785238  -0.48943722
   -0.03837892]
  [ 0.45731223 -1.09808    -2.139234   ... -1.4069853  -0.32891268
    0.4602864 ]
  ...
  [ 0.6976881   0.2515251  -0.02629205 ... -1.4450663  -0.17328389
   -0.7539515 ]
  [-0.0100829  -1.0530701  -1.5272577  ... -1.2557596  -1.0230237
   -0.09482554]
  [-1.0171427   0.21786046 -1.258674   ... -0.66509306 -0.20403881
   -0.45411202]]

 [[-0.2560048   0.01297291 -0.6495958  ... -2.2694743   0.6296554
    0.0541696 ]
  [-1.001227   -1.2824831   0.21143472 ... -2.0238378   0.3831914
    0.02885724]
  [-0.69083685 -0.9182259  -1.3843008  ... -1.1562237   0.48127314
    0.11296765]
  ...
  [-0.7620789  -0.53652936 -0.30214    ... -1.7766374   0.3351749
   -0.5287291 ]
  [-0.41872323 -1.0230036  -0.70452946 ... -1.0557156  -0.41289702
   -0.21295407]
  [-0.06315988 -0.6253144  -0.6342544  ... -0.94640225 -1.0022038
   -0.7379465 ]]

 [[-0.19802943  0.27728686 -1.8555619  ... -1.3240178  -0.9407664
    0.14993598]
  [ 0.03993949  0.26293856 -1.6493756  ... -0.5591016  -0.61810786
    0.46451846]
  [-1.3192503  -0.7833048  -0.08457345 ... -1.4012691  -1.2838948
    0.26721093]
  ...
  [-1.1363652  -0.3452591  -1.1286542  ... -0.24492458 -0.0342525
   -0.4959183 ]
  [-0.26522955 -0.50098014 -0.7568037  ... -0.7688988  -1.291214
   -1.2753637 ]
  [-0.6477462   0.18955947  0.15914796 ... -0.55584484 -0.70317924
    0.01292833]]], shape=(32, 10, 512), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 7-transformer_encoder_block.py
 
0/4 pts
7. Transformer Decoder Block
mandatory
Create a class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer:

Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
dm - the dimensionality of the model
h - the number of heads
hidden - the number of hidden units in the fully connected layer
drop_rate - the dropout rate
Sets the following public instance attributes:
mha1 - the first MultiHeadAttention layer
mha2 - the second MultiHeadAttention layer
dense_hidden - the hidden dense layer with hidden units and relu activation
dense_output - the output dense layer with dm units
layernorm1 - the first layer norm layer, with epsilon=1e-6
layernorm2 - the second layer norm layer, with epsilon=1e-6
layernorm3 - the third layer norm layer, with epsilon=1e-6
dropout1 - the first dropout layer
dropout2 - the second dropout layer
dropout3 - the third dropout layer
Public instance method def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
x - a tensor of shape (batch, target_seq_len, dm)containing the input to the decoder block
encoder_output - a tensor of shape (batch, input_seq_len, dm)containing the output of the encoder
training - a boolean to determine if the model is training
look_ahead_mask - the mask to be applied to the first multi head attention layer
padding_mask - the mask to be applied to the second multi head attention layer
Returns: a tensor of shape (batch, target_seq_len, dm) containing the block’s output
You should use MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention
$ cat 8-main.py
#!/usr/bin/env python3

import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

dblock = DecoderBlock(512, 8, 2048)
print(dblock.mha1)
print(dblock.mha2)
print(dblock.dense_hidden)
print(dblock.dense_output)
print(dblock.layernorm1)
print(dblock.layernorm2)
print(dblock.layernorm3)
print(dblock.dropout1)
print(dblock.dropout2)
print(dblock.dropout3)
x = tf.random.uniform((32, 15, 512))
hidden_states = tf.random.uniform((32, 10, 512))
output = dblock(x, hidden_states, False, None, None)
print(output)
$ ./8-main.py
<6-multihead_attention.MultiHeadAttention object at 0x0000018FB7B2CA60>
<6-multihead_attention.MultiHeadAttention object at 0x0000018FB7BB57C0>
<keras.layers.core.Dense object at 0x0000018FB7BC46D0>
<keras.layers.core.Dense object at 0x0000018FB7BC4B50>
<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000018FB7BC4E80>
<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000018FB7BD7070>
<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000018FB7BD71C0>
<keras.layers.core.Dropout object at 0x0000018FB7BD7430>
<keras.layers.core.Dropout object at 0x0000018FB7BD76A0>
<keras.layers.core.Dropout object at 0x0000018FB7BD7820>
tf.Tensor(
[[[-0.39095446  0.77960056  0.71250707 ...  0.8612595   2.81169
    1.1137545 ]
  [ 0.16502309  1.1479621   1.0586277  ...  1.2617519   1.9665152
    0.69792026]
  [ 0.39200962  1.3600402   0.8036952  ...  0.83441436  1.843133
    0.5498725 ]
  ...
  [ 0.27894798  0.9219343   1.3157915  ...  0.4419043   2.410698
    0.83392745]
  [ 0.01047665  1.005829    0.73982364 ...  1.55081     2.1921892
    1.5067849 ]
  [-0.4580258   1.9058013   1.4143255  ...  1.9357332   2.3433547
    0.9138516 ]]

 [[ 0.11028341  0.9745276   1.3977287  ...  0.993849    2.966307
    1.1493865 ]
  [ 0.04194991  1.2683535   1.3571532  ...  1.3767209   2.399031
    0.7990878 ]
  [-0.04151024  1.1736941   0.96711093 ...  0.731973    2.6785977
    0.6767394 ]
  ...
  [ 0.1797489   1.349999    1.2611121  ...  0.6840121   2.4925933
    1.2985908 ]
  [ 0.08293849  0.90350425  0.59766185 ...  0.5870124   2.6825192
    0.8823754 ]
  [-0.38025516  1.0896223   0.7785562  ...  0.23221144  2.6511576
    1.0880201 ]]

 [[ 0.21475774  1.9852333   1.1144592  ...  1.3977445   1.44852
    1.4127973 ]
  [ 0.17178851  1.4433401   1.3290592  ...  0.93091214  1.9300116
    1.2094498 ]
  [ 0.16188824  1.0856994   0.5858493  ...  0.6655169   2.0740108
    1.116149  ]
  ...
  [ 0.44694066  1.9452294   1.0224559  ...  1.2783171   2.3355215
    1.3663044 ]
  [ 0.56041795  2.2499764   0.515278   ...  0.8886313   2.0666676
    0.8075614 ]
  [-0.08543362  0.83389914  0.8914502  ...  1.807082    1.5164161
    0.9041297 ]]

 ...

 [[-0.25873432  1.0881183   1.60573    ...  0.20951171  2.2819428
    1.3666122 ]
  [ 0.65882176  1.343662    0.9826535  ...  1.1720676   2.5807462
    0.9180743 ]
  [-0.45900515  1.5224477   1.5491322  ...  0.704869    1.7304956
    1.1749669 ]
  ...
  [ 0.7684397   1.1356732   0.9601777  ...  0.5513381   2.4007556
    1.72613   ]
  [-0.4835616   1.6531435   1.1588017  ...  1.0223626   1.5151088
    0.2739644 ]
  [ 0.59966177  1.4279766   1.3221675  ...  1.2114371   1.8354752
    1.2011763 ]]

 [[ 0.4453847   1.7824427   1.0000769  ...  0.41760707  2.5735426
    0.64110714]
  [ 0.1672517   1.2706345   0.7132154  ...  1.2444379   2.1376762
    1.0232865 ]
  [ 0.00824916  1.0132847   1.401724   ...  0.9256221   2.3042448
    1.5273932 ]
  ...
  [ 0.5509354   1.6266351   1.5448375  ...  1.0948318   2.1406636
    1.5316169 ]
  [ 0.02547694  0.9789937   1.6408364  ...  1.0870866   1.8432375
    1.3281442 ]
  [ 0.55163044  1.853977    1.5687952  ...  1.0050749   1.8451209
    1.286485  ]]

 [[ 0.29871625  0.9412495   1.1800929  ...  1.4343112   2.1421087
    1.431788  ]
  [-0.52143335  1.9138077   1.4314412  ...  1.5200706   2.6795638
    0.9963926 ]
  [ 0.68740696  1.2671367   0.5891646  ...  0.95361394  2.422152
    1.438092  ]
  ...
  [ 0.6760997   1.498874    1.0875958  ...  0.81792563  1.7605903
    0.5104852 ]
  [ 0.15989637  0.67223537  0.34595788 ...  0.8347072   2.634362
    1.8129833 ]
  [-0.16527577  0.6887168   1.3223921  ...  1.161842    1.5979284
    1.5048783 ]]], shape=(32, 15, 512), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 8-transformer_decoder_block.py
 
0/4 pts
8. Transformer Encoder
mandatory
Create a class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer:

Class constructor def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
N - the number of blocks in the encoder
dm - the dimensionality of the model
h - the number of heads
hidden - the number of hidden units in the fully connected layer
input_vocab - the size of the input vocabulary
max_seq_len - the maximum sequence length possible
drop_rate - the dropout rate
Sets the following public instance attributes:
N - the number of blocks in the encoder
dm - the dimensionality of the model
embedding - the embedding layer for the inputs
positional_encoding - a numpy.ndarray of shape (max_seq_len, dm) containing the positional encodings
blocks - a list of length N containing all of the EncoderBlock‘s
dropout - the dropout layer, to be applied to the positional encodings
Public instance method call(self, x, training, mask):
x - a tensor of shape (batch, input_seq_len, dm)containing the input to the encoder
training - a boolean to determine if the model is training
mask - the mask to be applied for multi head attention
Returns: a tensor of shape (batch, input_seq_len, dm) containing the encoder output
You should use positional_encoding = __import__('4-positional_encoding').positional_encoding and EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
$ cat 9-main.py
#!/usr/bin/env python3

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder

encoder = Encoder(6, 512, 8, 2048, 10000, 1000)
print(encoder.dm)
print(encoder.N)
print(encoder.embedding)
print(encoder.positional_encoding)
print(encoder.blocks)
print(encoder.dropout)
x = tf.random.uniform((32, 10))
output = encoder(x, True, None)
print(output)
$ ./9-main.py
512
6
<keras.layers.embeddings.Embedding object at 0x000001D81D5C9E50>
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [-8.97967480e-01 -4.40061818e-01  4.26195541e-01 ...  9.94266169e-01
   1.03168405e-01  9.94663903e-01]
 [-8.55473152e-01  5.17847165e-01  9.86278111e-01 ...  9.94254673e-01
   1.03271514e-01  9.94653203e-01]
 [-2.64607527e-02  9.99649853e-01  6.97559894e-01 ...  9.94243164e-01
   1.03374623e-01  9.94642492e-01]]
ListWrapper([<7-transformer_encoder_block.EncoderBlock object at 0x000001D81D5ED100>, <7-transformer_encoder_block.EncoderBlock object at 0x000001D81DA6BEE0>, <7-transformer_encoder_block.EncoderBlock object at 0x000001D81DAB0D30>, <7-transformer_encoder_block.EncoderBlock object at 0x000001D81DABEB80>, <7-transformer_encoder_block.EncoderBlock object at 0x000001D81DACA9D0>, <7-transformer_encoder_block.EncoderBlock object at 0x000001D81DAD8820>])
<keras.layers.core.Dropout object at 0x000001D81DAE46A0>
tf.Tensor(
[[[-2.3610487  -1.1389643  -0.5434803  ...  0.35197693  1.2554014
    0.5808684 ]
  [-1.7336946  -1.6465555  -0.13855492 ...  0.21324947  2.2654455
    0.22438812]
  [-2.0648909  -2.2578015  -0.67286456 ...  0.35511485  2.2918317
    0.62210846]
  ...
  [-2.1527197  -1.1629148  -0.41166684 ...  0.11974051  1.3273743
    0.36166346]
  [-1.6019876  -0.9969617  -0.56715196 ...  0.36819518  2.7694468
    0.64485425]
  [-1.8037469  -1.6008706  -0.6957036  ...  1.1014873   2.2010543
    0.72015846]]

 [[-1.929273   -0.71806955 -0.33055028 ...  0.16928007  1.3144639
    0.41530192]
  [-2.1669066  -1.4770768  -0.12007697 ... -0.10891096  2.3498554
    0.21596757]
  [-2.2266057  -0.90345585 -0.5321402  ...  0.3268208   2.1824982
    0.36758614]
  ...
  [-1.9938254  -1.4193609  -0.28226373 ...  0.3240252   1.6510397
    0.4967216 ]
  [-1.9335259  -1.3976353  -0.22893184 ...  0.12964357  2.0164957
    1.0180595 ]
  [-2.557724   -0.92816556 -0.27856138 ...  0.1094092   1.3173697
   -0.15743211]]

 [[-1.3623643  -1.4590074   0.3846197  ...  0.5824262   2.0504875
    0.7992578 ]
  [-1.6867461  -0.84906834  0.36580923 ...  0.16000302  1.8226507
    0.55916643]
  [-2.2638314  -1.1281785  -0.31526992 ...  0.9251216   1.9877448
    0.4979302 ]
  ...
  [-1.7184831  -1.0086445   0.18233599 ...  1.1057613   1.5004982
   -0.1343396 ]
  [-2.0263698  -0.9899208   0.04571456 ...  0.21252109  2.3038921
    0.6646514 ]
  [-1.9500182  -0.80929697  0.6281491  ... -0.1607482   1.9314001
    0.37474895]]

 ...

 [[-1.1523433  -1.4773996  -0.63323665 ...  0.4644657   2.7011907
    1.1465223 ]
  [-2.0618405  -1.027301   -0.09930795 ...  0.22941993  2.424398
    0.40102696]
  [-2.0529535  -0.7129267  -0.54989403 ...  0.10707983  1.2631416
    0.61324275]
  ...
  [-1.5150424  -1.0365102  -0.8206318  ...  0.87440395  2.0604522
    0.7995123 ]
  [-1.9243824  -0.9179114  -0.5369217  ...  0.2736453   2.4756706
    0.97580653]
  [-1.6910121  -0.6832281  -0.36597902 ...  1.055163    2.4925356
    1.0925628 ]]

 [[-2.274539   -1.0737809  -0.22937298 ...  0.2279562   1.6521885
    0.48888496]
  [-2.2964559  -1.057159    0.20567828 ... -0.10671274  1.9343973
    0.46700588]
  [-1.8055079  -1.058835   -0.19558764 ... -0.27007443  1.9003891
    0.65915155]
  ...
  [-1.9021581  -1.1365409  -0.2105658  ...  0.48774236  1.9577066
    0.85815585]
  [-1.4764535  -1.3298213  -0.2127371  ...  0.04445247  2.0571017
    0.447816  ]
  [-1.1547536  -2.0336823  -0.25012437 ...  0.32559785  1.1743279
    1.0127293 ]]

 [[-2.215951   -1.3367128  -0.36792532 ...  0.15983997  2.1625319
    0.19500098]
  [-2.3084593  -1.5486513  -0.09061226 ...  0.14335689  2.153113
    0.80730885]
  [-2.4976354  -1.2277291  -0.23848003 ...  0.88765395  1.601327
    1.0631462 ]
  ...
  [-2.1868715  -0.59136045 -0.3066577  ...  0.74144876  1.6475624
    0.88572675]
  [-2.330432   -0.39941922 -0.09944266 ...  0.44615772  1.9127558
    0.74057215]
  [-1.6734713  -1.2846447  -0.29038197 ...  0.7843849   2.0464025
    0.3962947 ]]], shape=(32, 10, 512), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 9-transformer_encoder.py
 
0/4 pts
9. Transformer Decoder
mandatory
Create a class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer:

Class constructor def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
N - the number of blocks in the encoder
dm - the dimensionality of the model
h - the number of heads
hidden - the number of hidden units in the fully connected layer
target_vocab - the size of the target vocabulary
max_seq_len - the maximum sequence length possible
drop_rate - the dropout rate
Sets the following public instance attributes:
N - the number of blocks in the encoder
dm - the dimensionality of the model
embedding - the embedding layer for the targets
positional_encoding - a numpy.ndarray of shape (max_seq_len, dm) containing the positional encodings
blocks - a list of length N containing all of the DecoderBlock‘s
dropout - the dropout layer, to be applied to the positional encodings
Public instance method def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
x - a tensor of shape (batch, target_seq_len, dm)containing the input to the decoder
encoder_output - a tensor of shape (batch, input_seq_len, dm)containing the output of the encoder
training - a boolean to determine if the model is training
look_ahead_mask - the mask to be applied to the first multi head attention layer
padding_mask - the mask to be applied to the second multi head attention layer
Returns: a tensor of shape (batch, target_seq_len, dm) containing the decoder output
You should use positional_encoding = __import__('4-positional_encoding').positional_encoding and DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock
$ cat 10-main.py
#!/usr/bin/env python3

import tensorflow as tf
Decoder = __import__('10-transformer_decoder').Decoder

decoder = Decoder(6, 512, 8, 2048, 12000, 1500)
print(decoder.dm)
print(decoder.N)
print(decoder.embedding)
print(decoder.positional_encoding)
print(decoder.blocks)
print(decoder.dropout)
x = tf.random.uniform((32, 15))
hidden_states = tf.random.uniform((32, 10, 512))
output = decoder(x, hidden_states, True, None, None)
print(output)
$ ./10-main.py
512
6
<keras.layers.embeddings.Embedding object at 0x000002EACB1FCE50>
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.99516416e-01 -3.10955511e-02 -8.59441209e-01 ...  9.87088496e-01
   1.54561841e-01  9.87983116e-01]
 [ 5.13875021e-01 -8.57865061e-01 -6.94580536e-02 ...  9.87071278e-01
   1.54664258e-01  9.87967088e-01]
 [-4.44220699e-01 -8.95917390e-01  7.80301396e-01 ...  9.87054048e-01
   1.54766673e-01  9.87951050e-01]]
ListWrapper([<8-transformer_decoder_block.DecoderBlock object at 0x000002EACB221100>, <8-transformer_decoder_block.DecoderBlock object at 0x000002EACB8D7250>, <8-transformer_decoder_block.DecoderBlock object at 0x000002EACB8F63D0>, <8-transformer_decoder_block.DecoderBlock object at 0x000002EACB90A550>, <8-transformer_decoder_block.DecoderBlock object at 0x000002EACB91B6D0>, <8-transformer_decoder_block.DecoderBlock object at 0x000002EACB92F850>])
<keras.layers.core.Dropout object at 0x000002EACB943A30>
tf.Tensor(
[[[-1.2839141  -1.7186812  -0.16722882 ...  0.8535992  -0.3058408
   -0.07219017]
  [-1.0246873  -1.170053   -0.99896204 ...  0.9041854  -0.38787735
    0.01138113]
  [-0.91503954 -1.8266125  -0.9246276  ...  0.7101549  -0.25984824
    0.00641013]
  ...
  [-0.8930258  -1.3566624  -0.9167718  ...  0.903667    0.13016249
    0.34163323]
  [ 0.3480528  -0.39785624 -1.2213198  ...  1.3073617   0.0387878
   -0.11509471]
  [-0.9764983  -0.14782293 -0.6660392  ...  0.8612     -0.07568183
    0.850567  ]]

 [[-0.6303013  -1.6806108  -0.99345213 ...  0.41313174  0.13742425
    0.6340407 ]
  [ 0.60086066 -1.9879752  -0.36650416 ...  1.5152323  -0.16065322
    1.198124  ]
  [-0.55801415 -1.3498771  -0.9904915  ...  1.4153029   0.05561991
    0.32094598]
  ...
  [ 0.75655204 -0.12668519 -0.8338539  ...  0.66446066 -0.1949348
    0.43062454]
  [ 0.76843965 -1.5195376  -1.4645253  ...  0.9086696   0.06720471
    0.6674283 ]
  [-0.8310955  -1.7430267  -0.70096713 ...  1.0820472  -0.03668492
    1.1571299 ]]

 [[-0.76948565 -0.6904721  -0.08282385 ...  1.484637   -0.01079749
   -0.00923723]
  [-0.61875117 -1.0588669  -1.0004205  ...  1.6599276   0.11087392
    0.03695665]
  [-0.698759   -1.224315   -0.8011629  ...  1.0042421   0.13232033
    0.8384138 ]
  ...
  [-0.56767166 -0.67110103 -0.68728006 ...  1.3079039   0.41846672
    0.32642707]
  [-0.49781242 -0.3091485  -0.86568844 ...  1.2656546   0.13209444
    0.04619027]
  [-0.88079005 -1.2566048  -1.0898697  ...  1.717879   -0.15780629
    0.59974366]]

 ...

 [[-0.75548446 -1.2876992  -0.6482911  ...  1.028909   -0.3199176
    1.0764202 ]
  [-0.56363904 -1.6960151  -1.0017738  ...  1.1216183  -0.06809186
   -0.6220774 ]
  [-0.7739349  -1.0774732  -1.5090048  ...  1.0364398   0.35084996
    0.7928663 ]
  ...
  [-0.3840399  -1.8594748  -1.406895   ...  1.7091266   0.33853096
    0.05546568]
  [-0.61151963 -1.7188299  -0.46674782 ...  0.910305    0.73276395
    0.34628466]
  [-0.26632014 -1.4783959  -1.0045711  ...  0.60886633  0.17101993
    1.4237919 ]]

 [[-1.3076353  -0.4769247  -0.9589241  ...  1.6656599   0.2444006
    0.039077  ]
  [-0.43179825 -1.3804953  -1.2190187  ...  0.8791993   0.3585251
    1.3232248 ]
  [-0.8947926  -0.7076543  -1.7197332  ...  1.467966   -0.2327109
    0.8367135 ]
  ...
  [-1.1282836  -0.95806456 -1.3862897  ...  1.1811897  -0.00699645
    1.19855   ]
  [-1.0060518  -1.3372138  -1.3693043  ...  1.6093606   0.33184376
    0.92806995]
  [-0.47295657 -1.2602897  -0.31505173 ...  1.7068675   0.20713662
    1.35222   ]]

 [[-0.9844823  -1.6571848  -0.5853845  ...  1.1475037  -0.2165803
    0.30429295]
  [ 0.0491035  -1.2230506  -0.12719093 ...  0.6998839  -0.2387897
    0.32126707]
  [-1.0114121  -0.5240076  -0.72910136 ...  0.9303758  -0.06404482
   -0.09504666]
  ...
  [-1.0848765  -1.7569413  -0.15991753 ...  1.2040083  -0.38369745
    0.21836095]
  [-0.8922033  -1.2762529  -1.1504884  ...  1.5010442  -0.7252241
    0.02337704]
  [ 0.01927758 -1.7266833  -0.6291783  ...  0.83365756 -0.6204453
    0.03266546]]], shape=(32, 15, 512), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 10-transformer_decoder.py
 
0/4 pts
10. Transformer Network
mandatory
Create a class Transformer that inherits from tensorflow.keras.Model to create a transformer network:

Class constructor def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
N - the number of blocks in the encoder and decoder
dm - the dimensionality of the model
h - the number of heads
hidden - the number of hidden units in the fully connected layers
input_vocab - the size of the input vocabulary
target_vocab - the size of the target vocabulary
max_seq_input - the maximum sequence length possible for the input
max_seq_target - the maximum sequence length possible for the target
drop_rate - the dropout rate
Sets the following public instance attributes:
encoder - the encoder layer
decoder - the decoder layer
linear - a final Dense layer with target_vocab units
Public instance method def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
inputs - a tensor of shape (batch, input_seq_len)containing the inputs
target - a tensor of shape (batch, target_seq_len)containing the target
training - a boolean to determine if the model is training
encoder_mask - the padding mask to be applied to the encoder
look_ahead_mask - the look ahead mask to be applied to the decoder
decoder_mask - the padding mask to be applied to the decoder
Returns: a tensor of shape (batch, target_seq_len, target_vocab) containing the transformer output
You should use Encoder = __import__('9-transformer_encoder').Encoder and Decoder = __import__('10-transformer_decoder').Decoder
$ cat 11-main.py
#!/usr/bin/env python3

import tensorflow as tf
Transformer = __import__('11-transformer').Transformer

transformer = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)
print(transformer.encoder)
print(transformer.decoder)
print(transformer.linear)
x = tf.random.uniform((32, 10))
y = tf.random.uniform((32, 15))
output = transformer(x, y, True, None, None, None)
print(output)
$ ./11-main.py
<9-transformer_encoder.Encoder object at 0x00000218FEDB38E0>
<10-transformer_decoder.Decoder object at 0x000002189747F550>
<keras.layers.core.Dense object at 0x000002189762A280>
tf.Tensor(
[[[ 1.0607736   0.0242403  -0.09106813 ... -0.29175124 -0.16125385
    0.3864895 ]
  [ 0.95106304  0.01845607 -0.14091308 ... -0.21603766 -0.17990775
    0.2568456 ]
  [ 0.9478469  -0.02590757 -0.18830362 ... -0.26283994 -0.35723308
    0.1905604 ]
  ...
  [ 0.98177546  0.06318729  0.10035861 ... -0.12063442 -0.31148666
    0.30870026]
  [ 1.0850569   0.04290867 -0.10671776 ... -0.05698005 -0.15206534
    0.17607388]
  [ 0.83038425 -0.11482446 -0.21609192 ... -0.18703347 -0.31469542
    0.2715454 ]]

 [[ 0.6803826  -0.10935004 -0.19390443 ...  0.03436825 -0.36048222
    0.35624567]
  [ 0.59998506 -0.15790501 -0.08910711 ... -0.22004598 -0.22839934
    0.06497175]
  [ 0.6061491   0.09733439 -0.04216712 ... -0.07932705 -0.5246006
    0.2188019 ]
  ...
  [ 0.18082772  0.02340399 -0.19111699 ... -0.15864633 -0.1586476
    0.04199061]
  [ 0.5056264  -0.03862262 -0.10236847 ... -0.1429045  -0.39620322
    0.26531416]
  [ 0.60855716 -0.02337974  0.00709903 ... -0.11696491 -0.19906285
    0.22937746]]

 [[ 0.4756126   0.0421574  -0.18986051 ... -0.18700294 -0.4089745
    0.09693278]
  [ 0.43045658  0.09964329 -0.05200662 ... -0.2200033  -0.47586858
    0.30669567]
  [ 0.44368732 -0.09503576 -0.0334809  ... -0.22169325 -0.3907352
    0.41204658]
  ...
  [ 0.6086809  -0.16030277 -0.08422468 ... -0.2669319  -0.24752825
    0.292014  ]
  [ 0.61325645 -0.2813168   0.00505272 ... -0.3474741  -0.25893998
    0.05827443]
  [ 0.43180513 -0.06418268  0.11511327 ... -0.4831155  -0.2616658
    0.3848589 ]]

 ...

 [[ 0.5013967  -0.05441397 -0.09074385 ... -0.13244708 -0.1132496
    0.30883843]
  [ 0.66900104 -0.21307845 -0.03297749 ... -0.12286332 -0.18310091
    0.27747768]
  [ 0.5294878  -0.07785077 -0.17670675 ... -0.19229156 -0.11452198
    0.4405929 ]
  ...
  [ 0.50725204 -0.34826094 -0.05190882 ... -0.13106728 -0.12735105
    0.40158004]
  [ 0.71022266 -0.03333605  0.06456167 ...  0.02787622 -0.08124872
    0.46646377]
  [ 0.7886462  -0.23631734 -0.3200215  ... -0.05331544  0.05640198
    0.27362245]]

 [[ 0.6729588  -0.03007721  0.08204518 ... -0.2529254  -0.21775797
    0.11264613]
  [ 0.7462392  -0.06283787  0.00161131 ... -0.18139078 -0.19799335
    0.19990522]
  [ 0.6033629  -0.03189457  0.05997929 ... -0.11957973 -0.24930024
    0.18334061]
  ...
  [ 0.6706957  -0.18494028  0.10097492 ... -0.26747915 -0.23272479
    0.33512405]
  [ 0.7453947  -0.10429359 -0.14126898 ...  0.18224922 -0.19534475
    0.05128118]
  [ 0.5579367   0.05008543  0.04685662 ...  0.05295714  0.00513041
    0.32241058]]

 [[ 0.5590081   0.07322301 -0.07901593 ... -0.2180205  -0.26055804
    0.34265965]
  [ 0.5785861  -0.05857522  0.04306682 ... -0.28347975 -0.21662925
    0.32062888]
  [ 0.42751753 -0.11286587 -0.32851222 ... -0.21778783 -0.15483102
    0.1198507 ]
  ...
  [ 0.5392593   0.07121103 -0.17133081 ... -0.27553678 -0.317232
    0.31350407]
  [ 0.5041454  -0.12244039 -0.15222985 ... -0.23291877 -0.3511095
    0.19206154]
  [ 0.85592747  0.03772608 -0.10812019 ... -0.35709363 -0.19017766
    0.24399151]]], shape=(32, 15, 12000), dtype=float32)
$
Ignore the Warning messages in the output

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/attention
File: 11-transformer.py

https://web.stanford.edu/~jurafsky/slp3/old_dec21/10.pdf