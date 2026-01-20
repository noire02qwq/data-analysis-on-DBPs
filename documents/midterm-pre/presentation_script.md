# Midterm Presentation Script (Consolidated - 15 Minutes)

## Slide 1: Title Slide (0:00 - 0:30)
"Good morning/afternoon everyone. My name is Zhou Dafu. Today I will present my midterm progress on the **Development of Deep Learning Models for DBP Prediction in a Simulated Drinking Water Distribution Network**. This work is supervised by Prof. Hu Jiangyong."

## Slide 2: Outline (0:30 - 0:45)
"Basically, I'll cover three main parts:
1.  Introduction and Objectives.
2.  Methodology and Experimental Framework.
3.  Results and Conclusion."

# Section 1: Introduction & Objectives

## Slide 3: Background & DBP Formation (0:45 - 1:45)
"Disinfection By-products (DBPs) are a big issue. They happen when disinfectants mix with organic stuff in the water. We're mainly worried about **Trihalomethanes (THMs)**, **Haloacetic acids (HAAs)**, and **Haloacetonitriles (HANs)** because they're bad for health.
Our goal is to use Deep Learning for real-time prediction in a simulated drinking water network.
Chemically, chloramines react with organic precursors to create these harmful compounds, like Nitrosamines."

## Slide 4: Research Gap (1:45 - 2:30)
"So why do this research?
1.  **It's Complicated**: Old-school models just can't keep up with how complex and fast-changing water systems are.
2.  **Standard ML Fails**: Even regular Machine Learning, like Decision Trees, struggles with the complex time patterns in our data.
3.  **Lab Tests are Slow**: Checking water quality in a lab takes too long and costs too much.
4.  **The Opportunity**: We have sensors giving us data every few minutes. Deep Learning can use that for real-time control that other methods can't match."

## Slide 5: Objectives (2:30 - 3:00)
"Our objectives are:
1.  **Development of Deep Learning Models**: Comparing MLP, RNN, LSTM, and GRU.
2.  **Analyze Correlations (in progress)**: Investigating correlations between sensors and DBPs.
3.  **Develop Prediction Framework (in progress)**: Building a robust system for real-time monitoring."

# Section 2: Methodology & Experimental Framework

## Slide 6: System Description (3:00 - 3:30)
"We simulated a Water Distribution System with a Disinfection Tank, Retention Tank, and two Pipelines. This allows us to track water quality evolution."

## Slide 7: Data Acquisition (3:30 - 4:00)
"We utilize a comprehensive sensor network to capture a holistic view of the water quality.
We're monitoring core parameters that you'd expect, like **Total Residual Chlorine (TRC)** and **pH**, which give us the basic state of the water.
But we go further by tracking advanced indicators like **Total Organic Carbon (TOC)** and **Fluorescent Dissolved Organic Matter (fDOM)**. These are critical because they act as proxies for the organic 'fuel' that reacts with chlorine to form DBPs.
We capture this data at a high resolution—every 5 minutes. This creates a dense time-series dataset that allows us to catch even brief fluctuations in water quality that traditional daily sampling would completely miss."

## Slide 8: Data Preprocessing: Imputation (4:00 - 4:45)
"For cleaning up the data, the first challenge is handling missing values. Our sensors operate on an alternating cycle, which unfortunately leaves us with consistent 30-minute gaps in the data stream.
Standard linear interpolation—just drawing a straight line between points—is often 'too clean'. It artificially smooths out the data, removing the natural high-frequency noise that real sensors always have.
To solve this, we use **Stochastic Interpolation**. Basically, we fill the gap by estimating the underlying trend, but then we consciously inject a controlled amount of random noise back into that filled section. This noise isn't random processing junk; it's calibrated to match the local variance of the actual sensor readings. This ensures that our training data looks and feels like real sensor data, preventing the model from learning artificial smooth patterns that don't exist in reality."

## Slide 9: Data Preprocessing: Time Alignment (4:45 - 5:15)
"Then there's Time Alignment. Since water moves from the tank to the pipes, there's a delay. We fix this by shifting the data so we're always comparing the **same batch of water** as it flows through."

## Slide 10: Model Architectures (5:15 - 6:00)
"We tested four distinct deep learning architectures to see which one handles this complex chemical dynamics best.

First is the **Multilayer Perceptron (MLP)**. Think of this as our baseline. It's a standard feedforward network. We take a window of past data, flatten it into a single long vector, and feed it through. It treats the input as a static snapshot, without really understanding that 'time' is a sequence. It's powerful, but it lacks a sense of history.

Next, we tried the **Recurrent Neural Network (RNN)**. This is the classic architecture designed for sequences. It has a 'hidden state'—a memory that gets passed forward from one time step to the next. This allows it to remember previous inputs. However, standard RNNs struggle with a well-known problem called 'vanishing gradients'. Essentially, as you look further back in time, its memory gets fuzzy. It's good at remembering what happened 5 minutes ago, but terrible at remembering what happened an hour ago.

To fix that short-term memory issue, we use the **Long Short-Term Memory (LSTM)** network. This is a much more complex beast. Imagine a conveyor belt running through the network—that's its cell state, capable of carrying information over very long distances.
The LSTM uses three intelligent 'gates' to control this memory:
1.  A **Forget Gate** that decides what old info to throw away.
2.  An **Input Gate** that decides what new info is worth storing.
3.  An **Output Gate** that decides what to tell the next layer.
This mechanism allows it to learn complex, long-term dependencies, like how a spike in organic matter an hour ago might affect chlorine levels right now.

Finally, we tested the **Gated Recurrent Unit (GRU)**. You can think of this as a streamlined, more efficient cousin of the LSTM. It combines the forget and input gates into a single 'update gate'. It often achieves similar performance to LSTM but with a simpler architecture and faster training times.
We picked these recurrent models specifically because water quality isn't static—it's a story that unfolds over time, and these models are designed to read that story."

## Slide 11: Training Process (6:00 - 6:45)
"Here's how we train them.
We use **Mean Squared Error (MSE)** to measure mistakes.
We use the **Adam optimizer** because it adjusts itself.
To keep the model from just memorizing the data (overfitting), we use **Dropout** and **Early Stopping**—basically stopping when it stops getting better. We carefully split our data into Training, Validation, and Test sets."

## Slide 12: Hyperparameter Optimization (6:45 - 7:30)
"Tuning these models—finding the perfect learning rate, layer size, and dropout rate—is notoriously tricky. The search space is vast.
If we used a simple Grid Search, we'd essentially be blindly checking every single combination, which is incredibly slow and inefficient.
Instead, we used **Bayesian Optimization**. Think of this like using a metal detector that beeps louder as you get closer to the treasure, rather than digging random holes in the sand.
It builds a probabilistic model of how the hyperparameters affect the model's accuracy. With every experiment it runs, it updates this internal map, becoming smarter about where to look next. It balances 'exploration'—trying wild new settings just in case—with 'exploitation'—refining the settings that already look promising. This allowed us to find optimal configurations with a fraction of the computational cost."

## Slide 13: Algorithm Optimizing (Rate of Change) (7:30 - 8:15)
"We optimized our algorithms with two strategies.
First, the **Normalized Rate of Change**.
Here's the problem: if you look at the raw data, the downstream concentration is usually just the upstream concentration minus a tiny bit of decay. A model can get 99% accuracy just by copying the input. It's 'cheating' without learning the chemistry.
By predicting the 'Rate'—the percentage change—we force the model to ignore the absolute numbers and focus purely on the *change* itself. We are essentially forcing the neural network to learn the *kinetics* of the chemical reaction occurring inside the pipe. It transforms the problem from 'predict the value' to 'predict the reaction speed', which is much more stable and generalizable."

## Slide 14: Algorithm Optimizing (Decoupled Model) (8:15 - 9:00)
"Second, **Decoupled Modeling**.
We found that Total Residual Chlorine (TRC) behaves very differently from stable parameters like pH or TOC. Chlorine is highly reactive and decays rapidly based on temperature and organic load. pH and TOC, on the other hand, are relatively conservative tracers.
When we tried to force one single model to predict all of them at once, it got confused. The loss function was pulled in different directions.
So, we engaged in **Decoupled Modeling**. We built a dedicated **TRC Model** exclusively focused on the difficult task of chlorine decay. Then, we built a separate **Other Parameter Model** to handle the stable parameters. This allows us to use a more complex, sensitive architecture for the chlorine model, while keeping a simpler one for the others. It's about using the right tool for the job."

# Section 3: Results & Conclusion

## Slide 15: Performance Metrics (9:00 - 9:45)
"Let's look at the results.
The table here shows the Mean Squared Error (MSE) for our different experiments. Lower is better.
For **Total Residual Chlorine (TRC)**, look at the first column. The **LSTM with Rate regression** achieves the lowest error of 0.0041. This validates our hypothesis: the LSTM's ability to remember long-term history, combined with the 'Rate' strategy focusing on kinetics, creates the best predictor for reactive chemicals.
However, for stable parameters like **pH** and **TOC**, the story flips. The **GRU with Value regression** comes out on top. Since these values don't change much as they flow through the pipe, the simpler GRU architecture predicting the direct value is more stable and accurate than trying to predict a near-zero rate of change."

## Slide 16: Prediction Visualization (9:45 - 10:15)
"The plots confirm that our best models closely track the actual sensor trends."

## Slide 17: Conclusion & Future Work (10:15 - 11:15)
"So, the key takeaways:
1.  **It Works**: Neural Networks handle this complex water data really well.
2.  **Time Matters**: GRU and LSTM beat the others, showing that tracking time patterns is crucial.
3.  **Strategy is Key**: Predicting 'Rate' for fast-changing things (Chlorine) and 'Value' for stable things works best.

Future Work:
We will focus on a **Second-Stage Model** to map these predicted water quality references (TRC, pH, TOC) to actual DBP concentrations (THMs, HAAs)."

## Slide 18: Q&A (11:15 - 15:00)
"Thank you. Questions?"
