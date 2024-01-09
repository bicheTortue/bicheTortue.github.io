---
layout: page
title: projects.titles.masterthesis
description: projects.descriptions.masterthesis
img: assets/projects/mthesis/lstmCircuit.svg
importance: 1
category: work
#github: https://github.com/bicheTortue/MSc-thesis
related_publications: gru
---

*[HTML]: Hyper Text Markup Language
*[AI]: Artificial Intelligence
*[ASIC]: Application-Specific Integrated Circuit
*[C. elegans]: Caenorhabditis elegans
*[CIFG]: Coupled Input-Forget Gate
*[CMOS]: Complementary metal-oxide-semiconductor
*[CPU]: Central Processing Unit
*[FCM]: Forward Crawling Motion
*[FGR]: Full Gate Recurrence
*[FPGA]: Field Programmable Gate Array
*[GPT]: Generative Pre-trained Transformer
*[GPU]: Graphics Processing Unit
*[GRU]: Gated Recurrent Unit
*[INESC]: Instituto de Engenharia de Sistemas e Computadores
*[IST]: Instituto Superior Tecnico
*[LIDM]: Linear Ion Drift Model
*[LLaMA]: Large Language Model Meta AI
*[LLM]: Large Language Model
*[LSTM]: Long Short-Term Memory
*[MSE]: Mean Square Error
*[NFG]: No Forget Gate
*[NIAF]: No Input Activation Function
*[NIG]: No Input Gate
*[NN]: Neural Network
*[NOAF]: No Output Activation Function
*[NOG]: No Output Gate
*[NP]: No Peephole
*[opAmp]: operational amplifier
*[RMSE]: Root Mean Square Error
*[RNN]: Recurrent Neural Network
*[STBM]: Simmons Tunnel Barrier Model
*[tanh]: hyperbolic tangent
*[TEAM]: ThrEshold Adaptive Memristor Model
*[VMM]: Vector Matrix Multiplication
*[VR]: Virtual Reality
*[VTEAM]: Voltage ThrEshold Adaptive Memristor Model

# Memristor-based recurrent modules for neural computing

## Abstract

In this work we propose an analog structure for memristor based recurrent modules targeting neural computing. The system is fully analog and implements a working LSTM circuit block and a work in progress GRU circuit block. Both of those blocks contain memristors to be used as weights in a analog VMM capable circuit. These circuit blocks allow to run very fast computation of RNNs of any size, in a relatively small integrated circuit. As part of the LSTM and GRU blocks, an analog activation function circuit was designed. This specific circuit is capable of reproducing sigmoid and tanh like functions, with similar shapes and the same output ranges. The work also include the implementation of a memory cell used to store an analog value for a short period of time. The LSTM block can be serialized or not with the ability to choose the level of serialization. Serializing the system allows to save onChip area at the cost of execution time. To the author's knwoledge this is the first analog implementation of the behavior of C. elegans using the LSTM block. Such an analog system provides ground for real time implementation of nervous systems.

### Keywords

Analog neural computing, Embedded neural computing, Memristor-based recurrent modules, Analog LSTM, Analog GRU

## Introduction

AI are one of the main research domain in computer sciences and is used in other scientific domains. One great application of AI, is in embedded systems. Lots of small electronic devices around us could benefit us. **Google** and **Apple** are pushing AI in phones with their in house designed ARM processors having, very advertised, tensor cores.
Being able to run low powered AI is thus one of the biggest computer objective of the deceny.
Analog computers are known to be a low powered technology that most importantly give almost instant results. An analog ASIC could then be fabricated for specific embedded applications. This would definitely improve digital technologies in terms of time, and maybe in power consumption as well, these being two prerequisite for embedded system use.
For this reason this work aims at creating a functioning software simulation of analog circuit capable of running AI. The thesis will focus on the software simulation, the first step of the work in order to have a fully working chip able to run AI algorithms.


## State of the art

### Recurrent Neural Networks (RNN)

RNN are a family of neural networks that differentiate themselves by having feedback connections. It is often used with sequences of data \cite{rnn}, sometimes, with varying amount of input data. RNNs are used for handwriting recognition and language translation \cite{gru}.

The feedback connection is caracterized by the hidden state vector ($\symvh_t$) which serves as the output and as part of the input for the next time step, a RNN is defined as \cref{eq:rnn}.

\begin{equation}\label{eq:rnn}
\symvh_t=f(\symvx_t,\symvh_{t-1})
\end{equation}


The simple version of the RNN is defined by \cref{eq:srnn}.

\begin{equation}\label{eq:srnn}
\symvh_t=tanh([\symvx_t,\symvh_{t-1}]\cdot \symmw + \symvb)
\end{equation}

Where (\symmw,\symvb) are the pair of weights matrix and bias vectors. $\symvx_t$ is the input vector and $\symvh_t$ is the hidden state vector.

The graphical representation of the simple RNN cell is shown in \cref{fig:rnnCell}.

\begin{figure}[t]
\centering
\begin{minipage}{\columnwidth}
\subfloat[Simple \acs{RNN} cell\label{fig:rnnCell}]{\includesvg[width=\columnwidth]{rnn/rnnCell.svg}}
\end{minipage}
\begin{minipage}{\columnwidth}
\subfloat[Legend\label{leg:cells}]{\includesvg[width=\columnwidth,pretex=\footnotesize]{cellsLegend.svg}}
\end{minipage}
\caption{}
\end{figure}

\subsection{Long Short Term Memory (LSTM)}

LSTMs are a type of RNN used to solve the vanishing gradiant problem \cite{firstLSTM}. It was improved a few times before becoming the modern LSTM \cite{improvLSTM}. The LSTM differs from a simple RNN because of its second feedback variable being the cell state ($\symvc_t$).

The LSTM contains four activated gates, that each serve its own purpose. The input gate \cref{eq:inputG} controls whether the cell state is updated, the forget gate \cref{eq:forgetG} that determines how the current cell state is affected by the old cell state, the output gate \cref{eq:outputG} that controls how much the hidden state is affected by the cell state. The candidate cell state gate \cref{eq:candCell} computes the change in the future cell state.

\begin{equation}\label{eq:inputG}
\symvi_t= \sigma([\symvx_{t_1},\symvh_{t-1}]\cdot \symmw_i + \symvb_i)
\end{equation}
\begin{equation}\label{eq:forgetG}
\symvf_t= \sigma([\symvx_{t_1},\symvh_{t-1}]\cdot \symmw_f + \symvb_f)
\end{equation}
\begin{equation}\label{eq:outputG}
\symvo_t= \sigma([\symvx_{t_1},\symvh_{t-1}]\cdot \symmw_o + \symvb_o)
\end{equation}
\begin{equation}\label{eq:candCell}
\symvct_t=tanh([\symvx_{t_1},\symvh_{t-1}]\cdot \symmw_c+ \symvb_c)
\end{equation}

The next part of the LSTM consist of computing the new cell state in the following equation :

\begin{equation}\label{eq:cellS}
\symvc_t=\symvf_t\odot \symvc_{t-1} + \symvi_t \odot \symvct_t
\end{equation}

The hidden state is then determined using the current cell state \cref{eq:hiddenS}.

\begin{equation}\label{eq:hiddenS}
\symvh_t=\symvo_t\odot tanh(\symvc_t)
\end{equation}

Where ($\symmw_i$,$\symvb_i$), ($\symmw_f$,$\symvb_f$), ($\symmw_o$,$\symvb_o$) and ($\symmw_c$,$\symvb_c$) are the pair of weights matrixes and bias vectors for the input, forget, output and candidate cell gates respectively. $\symvx_t$ is the input vector and $\symvh_t$ is the hidden state vector.

The graphical representation of an LSTM cell is found in \cref{fig:lstmCell}.

\begin{figure}[t]
\centering
\includesvg[width=\columnwidth]{lstm/lstmCell.svg}
\caption{\acs{LSTM} cell, adapted from \cite{wikiLSTM}\label{fig:lstmCell}}
\end{figure}

\subsection{Gated Recurrent Units (GRU)}

GRUs are another type of RNN. It is also known to reduce the effect of the vanishing gradient problem. It was first introduced to improve translation techniques \cite{gru}.

The \ac{GRU} is very often compared to the \ac{LSTM}, being sometimes assimilated as a type of \ac{LSTM} \cite{nbLSTM}. Their performance was found to be very similar in most situations \cite{gruVSlstm}, making those two types of \acp{RNN} coexistant in the modern machine learning world.

There are two versions of the \ac{GRU}, both are found on the internet, they are known as the encoder and decoder version \cite{gru}. They were originally designed to encode the message to translate and then decode in the translation. PyTorch only supports the decoder version \cite{gruPyTorch}, while the Keras library supports both \cite{gruKeras} chosen by changing an argument. This work only uses the encoder version, for that reason, it will be the only one described.

It contains an update gate \cref{eq:updateG}, a reset gate \cref{eq:resetG}, a candidate activation gate \cref{eq:candActivG}. The hidden state is then computed \cref{eq:gruHidG} using the previous results.

\begin{equation}\label{eq:updateG}
\symvz_t= \sigma([\symvx_t,\symvh_{t-1}] \cdot \symmw_z + \symvb_z)
\end{equation}
\begin{equation}\label{eq:resetG}
\symvr_t= \sigma([\symvx_t,\symvh_{t-1}] \cdot \symmw_r + \symvb_r)
\end{equation}
\begin{equation}\label{eq:candActivG}
\symvhh_t=tanh(\symvx_t\cdot \symmw_{hx}+(\symvr_t\odot\symvh_{t-1}) \cdot \symmw_{hh} + \symvb_h)
\end{equation}
\begin{equation}\label{eq:gruHidG}
\symvh_t=(\vunit-\symvz_t)\odot \symvh_{t-1} + \symvz_t\odot \symvhh_t
\end{equation}

Where ($\symmw_z$,$\symvb_z$), ($\symmw_r$,$\symvb_r$),($\symmw_{hx}$,$\symmw_{hh}$,$\symvb_h$) are the weights matrixes and bias vectors for the update, reset and candidate activation gates respectively.

A visual representation of the encoder \ac{GRU} cell is available in \cref{fig:encoderGruCell}.

\begin{figure}[t]
\centering
\includesvg[width=\columnwidth]{gru/encoderCell.svg}
\caption{Encoder \acs{GRU} cell, legend in \cref{leg:cells}\label{fig:encoderGruCell}}
\end{figure}

\subsection{Memristors}

Memristors are the lesser known fourth fundamental passive component of electronics, among resistors, capacitors and inductor.
It was first theorized in 1971 as a missing fundamental component in \cite{TheoMemristor}. The name comes from the blend of \textit{memory} and \textit{resistance}.
The missing component linking the four fundamental circuit variables, voltage (\symv), charge (\symq), current (\symi) and flux (\symphi). \Cref{fig:fundComp} shows the four fundamental variables are on each side of the square, with the ones on opposite sides being linked by the following equations :

\begin{equation}
d\symphi = \symv\cdot d\symt
\end{equation}

\begin{equation}
d\symq = \symi\cdot d\symt
\end{equation}

Resistors, capacitors and inductors were already very established and well known components, so it was theorized that a fourth device should then exist to physically link flux (\symphi) and charge (\symq).  The flux in this case is not a magnetic flux and is defined as such : $ d\symphi=\symv\cdot d\symt \Rightarrow \symphi =  \int \symv \,d\symt  $.

The component stayed theoretical until 2008 when it was implemented in a physical device for the first time \cite{memristorFab}. It took 37 years to have an actual working device.

An extension to the memristor, reffered to as the memristive device, was theorized in 1976 \cite{memrestiveDev}. The difference between the memristor and the memristive devices is its internal behavior. Memristive devices are commonly referred to as memristors as well.

\begin{figure}[t]
\centering
\includesvg[width=0.65\columnwidth,pretex=\tiny]{memristor/memristor}
\caption{Fundamental passive components, adapted from \cite{memWiki}}
\label{fig:fundComp}
\end{figure}

A memristor is used for its ability to change its internal resistance based on the current that flowed through it.

\subsection{Memristors Crossbar Arrays}

Setting memristors in a crossbar array allows to perform analog Vector Matrix Multiplication (VMM), also called Multiply and Accumulate. \Cref{fig:crossbar} shows what a typical crossbar array looks like.

\begin{figure}[b]
\centering
\includesvg[width=.6\columnwidth,pretex=\small]{crossbar/crossbar}
\caption{Crossbar array schematics, inspired from \cite{xbarFigures}}
\label{fig:crossbar}
\end{figure}

The circuit uses physical properties of electrical systems to perform analog computation. The following part will focus only on the circuit node in \cref{fig:crossNode}.

\begin{figure}[t]
\centering
\includesvg[wdith=0.4\columnwidth,pretex=\scriptsize]{crossbar/node}
%\def\svgheigth{3.5cm}
%\input{crossbar/node.pdf_tex}
\caption{Memristor crossbar node of the $k^{th}$ line and $j^{th}$ column}
\label{fig:crossNode}
\end{figure}

A voltage is applied on the $k^{th}$ line, and because every column is virtually grounded, the voltage applied to the memristor, with resistance $R_i$, is $V_i$. By applying Ohm's law, we know that the current flowing into the memristor ($\symi_{k}$) is bound by the following equation :

\begin{equation}
\symv_k = \symR\cdot \symi_{k} \Rightarrow \symi_{k} = \symv_k\cdot (\frac{1}{\symR_k})= \symv_k\cdot \symG_k
\end{equation}
With $ \symG_k$ being the conductance of memristor, defined as $ \symG_k=\frac{1}{\symR_k}$.

This line then joins the column where a current of $\symi_{j,k-1}$ is flowing, then according to Kirchhoff's current law the resulting current is :
\begin{equation}
\symi_{j,k} = \symi_{j,k-1}+\symi_{k} = \symi_{j,k-1} + \symv_k\cdot \symG_k
\end{equation}
Unfolding the equations will give the current at the bottom of the column, for example, the current at the bottom of the first column in \cref{fig:crossbar} is :
\begin{equation}
\symi_1= \symG_1\cdot \symv_1 +  \symG_2\cdot \symv_2 +  \symG_3\cdot \symv_3
\end{equation}
With $\symG_1$, $\symG_2$ and $\symG_3$ being the conductance of the 3 memristors in the first column.





    Every project has a beautiful feature showcase page.
    It's easy to include images in a flexible 3-column grid format.
    Make your photos 1/3, 2/3, or full width.

    To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

    <div class="row">
    <div class="col-sm mt-3 mt-md-0">
{% include figure.html path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
<div class="caption">
Test de caption
</div>
</div>
<div class="col-sm mt-3 mt-md-0">
{% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
<div class="col-sm mt-3 mt-md-0">
{% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
<div class="col-sm mt-3 mt-md-0">
{% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, *bled* for your project, and then... you reveal its glory in the next row of images.


<div class="row justify-content-sm-center">
<div class="col-sm-8 mt-3 mt-md-0">
{% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
<div class="col-sm-4 mt-3 mt-md-0">
{% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
<div class="col-sm-8 mt-3 mt-md-0">
{% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
<div class="col-sm-4 mt-3 mt-md-0">
{% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
</div>
```
{% endraw %}
