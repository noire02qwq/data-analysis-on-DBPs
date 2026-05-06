# Final Presentation Script (20 Minutes)

*Note: This script provides the spoken narrative for each slide. The presentation aims for a total duration of 20 minutes, allocating roughly 40-45 seconds per slide, and incorporating a 1-minute 30-second live video demonstration. The language is conversational but firmly rooted in environmental engineering and water chemistry.*

---

## Slide 1: Title Slide (0:00 - 0:30)
"Hello everyone. I'm Zhou Dafu. Today, I'll be presenting our final report on using Machine Learning to predict Disinfection By-products. I'd like to extend my gratitude to Professor Hu Jiangyong for supervising this research. Thank you all for joining."

## Slide 2: Outline (0:30 - 1:00)
"Here's a brief overview of our talk today. First, I'll discuss the background of DBPs and the specific engineering challenges we face. Then, I'll walk through our methodology, showing how we match AI architectures to physical reactor kinetics and hydraulics. Next, I'll demonstrate a practical software tool we built for plant operators. Finally, we'll dive into the results from a strictly environmental engineering perspective, followed by our future roadmap."

---
# Section 1: Introduction & Objectives

## Slide 3: Background & DBP Formation (1:00 - 1:50)
"Let's start with the core issue. Chlorine disinfection is vital, but it reacts with Natural Organic Matter to create hazardous Disinfection By-products like THMs and Nitrosamines. Chemically, in an alkaline water supply, monochloramines form and react with nitrogen precursors to generate these DBPs. Our goal is to leverage real-time sensor data—like Total Residual Chlorine, pH, and TOC—to predict and proactively control these harmful formations before they reach the consumer."

## Slide 4: Research Gap & Limitations of Traditional Approaches (1:50 - 2:30)
"Why are we turning to Deep Learning instead of traditional methods? Well, traditional mechanistic models are heavily constrained by rigid physical assumptions. They often fail to capture the highly dynamic, non-linear hydraulic conditions of real-world distribution networks. On top of that, standard lab testing is slow and expensive. Deep Learning offers a powerful alternative: it can autonomously learn the complex temporal relationships directly from high-frequency sensor data, without us manually defining the physics."

## Slide 5: Fine-Tuning Strategy for Reactor Generalization (2:30 - 3:15)
"In environmental engineering, a major challenge is transferability. A water distribution network spans vastly different conditions—varying temperatures, different water sources. It's wildly inefficient to train a brand-new model from scratch for every single physical change. 
Instead, we focus on exploring 'Reactor Generalization'. We introduce Fine-Tuning techniques to see if we can take the foundational chemical kinetics learned in a baseline reactor, and cleanly transfer that knowledge to new, similar reactors operating under different boundary conditions."

## Slide 6: Objectives (3:15 - 4:00)
"To summarize our goals: First, we aim to evaluate how well standard GBDTs and advanced Neural Networks can handle complex, continuous water quality time-series data. Second, we want to investigate the generalization limits of these models—specifically, how well can they transfer knowledge across temperature shifts and different water sources. Finally, we want to bridge the gap between academic research and practical operation by developing a production-ready GUI."

---
# Section 2: Methodology

## Slide 7: System Description (4:00 - 4:30)
"Moving on to our methodology. We utilized a simulated Water Distribution System comprising a Disinfection Tank, a Retention Tank, and two downstream pipelines. We deployed sensors across these key stages to track the water chemistry at dense 5-minute intervals."

## Slide 8: Data Preprocessing: Stochastic Imputation (4:30 - 5:15)
"Real-world sensor data is rarely perfect. Because our sensors alternated between the two pipelines, we faced consistent 30-minute data gaps. To solve this without distorting the physical reality of the data, we used Stochastic Imputation. We didn't just draw flat lines; we filled the gaps using local regressions and injected random noise that matched the variance of the real sensors. This completely preserves the natural physical fluctuations of the water flow."

## Slide 9: Hydraulic Retention Time (HRT) & Temporal Dynamics (5:15 - 6:00)
"Now, let's look at the physical flow. Water doesn't jump from the Disinfection Tank to the end of the pipeline instantly. It physically takes hours to travel that distance. We call this the Hydraulic Retention Time, or HRT. 
During this entire transit, chlorine decay and DBP formation are happening continuously. Traditional static modeling completely ignores this dynamic flow. By using a time-series sliding window approach, we are physically tracking that 'water parcel' as it ages and reacts along the pipe."

## Slide 10: Model Architectures Overview (6:00 - 6:30)
"For our modeling phase, we compared two primary families of algorithms. The first is GBDTs—like XGBoost and LightGBM—which are fast, tree-based models that partition data into decision boundaries. The second family is Neural Networks. This includes classic recursive models like LSTM and GRU, which are naturally suited for continuous time-series, as well as state-of-the-art architectures like the multi-head attention Transformer and the Mamba State-Space model."

## Slide 11: Neural Network Architectures (6:30 - 7:00)
"Here, you can see the structural diagrams of the Neural Networks we tested. On the top row, we have the simpler, classic models: the standard MLP, the basic RNN, and the highly-gated LSTM. On the bottom row, we have the GRU, the Transformer Encoder, and Mamba. Each of these handles sequential data differently, but they all share a common trait: they use layers of non-linear math to approximate complex, continuous changes over time."

## Slide 12: Why Transformer is Suitable for Water Reactors (7:00 - 7:45)
"I want to specifically highlight the Transformer. Its core is the Attention mechanism—shown by the mathematical formula here. 
Why does this matter for environmental engineering? Because water reactors and pipelines operate on highly cyclical, periodic schedules. The physics and chemistry happening in a pipe today are structurally very similar to what happened during the same cycle yesterday. The Attention mechanism calculates correlations across the entire history simultaneously. It naturally 'attends' to these recurring cyclical physical states far more effectively than models that just read data step-by-step."

## Slide 13: Model Training Workflow (7:45 - 8:15)
"This slide maps out our overall model training workflow. It illustrates the pipeline from raw sensor readings, through our data imputation and strict chronological splitting, down to the Bayesian hyperparameter tuning and final model evaluation."

## Slide 14: Fine-Tuning Workflow (8:15 - 8:45)
"For our transferability tests, we established two baseline Transformer models at $29^{\circ}\text{C}$ for our two distinct water sources. We then applied fine-tuning strategies to adapt them to the $35^{\circ}\text{C}$ datasets. This isolated approach allows us to observe exactly how the models learn the new kinetic reactions driven purely by the shift in temperature."

---
# Section 3: GUI Development

## Slide 15: Frontend Software Application (8:45 - 9:30)
"We recognize that advanced AI is useless to an operator if it isn't accessible. So, we built a fully interactive Graphical User Interface. We used Vite, React, and Electron to create a fast, clean desktop application. This software allows plant operators to effortlessly run our complex Python machine-learning backend on their local machines, directly applying our research to daily operations."

## Slide 16: Live Demonstration (Video) (9:30 - 11:00)
"To demonstrate this, I'll now play a brief 1-minute and 30-second video walkthrough of the software. You'll see an operator loading reactor data, initiating a model, and viewing real-time quality alerts."
*(Play Video - Pause Presentation)*

---
# Section 4: Results & Conclusion

## Slide 17: Chemical Kinetics: First-Order Decay in Pipelines (11:00 - 11:45)
"Welcome back. Before we dive into the numbers, let's talk about the actual chemistry happening in the pipes. 
The decay of Total Residual Chlorine from the upstream tanks down to the pipelines mathematically follows a first-order chemical reaction rate equation: $C_t = C_0 e^{-kt}$. 
This isn't a static equation. That reaction rate constant, $k$, is being dynamically influenced in real-time by environmental conditions like temperature, pH, and flow velocity. This creates a very continuous, highly non-linear exponential decay curve."

## Slide 18: Algorithm Expressiveness: GBDT vs Neural Networks (11:45 - 12:30)
"So, how do our algorithms fit this reality? 
Neural Networks use nested non-linear activation functions. Mathematically, they are naturally structured to map and approximate these smooth, continuous exponential transformations.
GBDTs, however, work entirely differently. They construct orthogonal, discontinuous step functions. They chop the data space into rigid boxes. Physically, they lack the mathematical expression power to extrapolate a smooth chemical decay curve beyond the discrete boundaries they saw during training."

## Slide 19: Evaluation Environment & GBDT Metrics (12:30 - 13:15)
"Keeping that in mind, let's look at the GBDT metrics. XGBoost performed numerically the best here. However, as noted at the bottom, while it works well within the strict bounds of its training data, it struggles conceptually. Because it relies on step-wise splitting, it cannot model actual temporal flow dynamics, and it physically cannot predict a chemical concentration higher or lower than what it has historically seen."

## Slide 20: GBDT Models: Visualization (13:15 - 13:45)
"This limitation is visibly obvious in the plots. While XGBoost on the left tracks the data decently, LightGBM and CatBoost are extremely messy. The predictions look jagged and stepped, which perfectly reflects their tree-based, non-continuous nature. This isn't how water chemistry actually behaves."

## Slide 21: Neural Network Metrics (13:45 - 14:30)
"When we switch to Neural Networks, the metrics improve. The LSTM achieved the lowest error overall. The LSTM's gated memory is perfectly suited for tracking these continuous, moderate-length sequential decay patterns in our limited dataset. 
However, it's crucial to note that the Transformer has significantly greater mathematical capacity. If we had massive volumes of parallel data from hundreds of different reactors, the Transformer's potential to map global environmental patterns would far exceed the classical LSTM."

## Slide 22: Neural Networks: Visualization (Part 1) (14:30 - 15:00)
"Looking at the simpler neural networks—MLP, RNN, and GRU—the predictions are much smoother and more continuous than the GBDTs. They naturally fit the smooth curves of chemical decay. The GRU on the right is exceptionally accurate, nearly matching the LSTM."

## Slide 23: Neural Networks: Visualization (Part 2) (15:00 - 15:45)
"And here are the LSTM, Transformer, and Mamba. The LSTM on the left provides the tightest fit for this specific reactor setup. The Transformer and Mamba capture the main trends successfully but show slightly more local volatility. So, for localized, single-reactor time-series regression, classic recurrent models currently hold a slight edge in raw precision."

## Slide 24: Transformer Fine-Tuning Performance (15:45 - 16:45)
"Finally, let's look at our Fine-Tuning generalization. We moved from $29^{\circ}\text{C}$ to $35^{\circ}\text{C}$. For the noisier LSWW water source, Partial Fine-Tuning was the clear winner. 
Why does this make sense environmentally? In a neural network, different layers act like perceptrons for different classes of environmental conditions—like pH or conductivity. By freezing the deep layers, we lock in the foundational physical and chemical relationships the model already learned. We only retrain the shallow, specific neurons sensitive to the 'temperature' shift. This isolates the environmental variable without corrupting the model's core physical knowledge."

## Slide 25: Conclusion (16:45 - 17:45)
"To summarize our key engineering takeaways:
First, Neural Networks, due to their continuous activation functions, are fundamentally superior to GBDTs for modeling kinetic environmental transformations like chemical decay.
Second, we proved that Transformer architectures offer exceptional transferability. By using Partial Fine-Tuning, we can effectively isolate specific environmental shifts—like temperature—and deploy the model to new reactors.
Third, our GUI application successfully translates this advanced kinetic modeling into a practical tool for the industry."

## Slide 26: Future Work (17:45 - 18:30)
"For our future work, we have two primary goals. First, we plan to integrate SHAP values. This will allow us to open the 'black box' and scientifically interpret exactly which physical sensors are biologically or chemically driving the model's decisions at any given time.
Second, we want to build a second-stage model that maps our accurately predicted proxy variables directly to lab-measured THM and HAA concentrations."

## Slide 27: Conceptual Roadmap (18:30 - 19:15)
"This conceptual roadmap illustrates that future integration, showing how our current continuous sensor predictions will feed directly into the final physical mapping of harmful DBPs."

## Slide 28: Q&A (19:15 - 20:00)
"That concludes my presentation. Thank you very much for your attention. I would now be happy to take any questions."
