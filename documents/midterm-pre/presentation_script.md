# Midterm Presentation Script (Consolidated - 15 Minutes)

## Slide 1: Title Slide (0:00 - 0:30)
"Good morning/afternoon everyone. My name is Zhou Dafu. Today I will present my midterm progress on the **Development of Deep Learning Models for DBP Prediction in a Simulated Drinking Water Distribution Network**. This work is supervised by Prof. Hu Jiangyong and mentored by Mr. Sun Yuanpeng."

## Slide 2: Outline (0:30 - 0:45)
"My presentation is divided into three main sections:
1.  Introduction and Objectives.
2.  Methodology and Experimental Framework.
3.  Results and Conclusion."

# Section 1: Introduction & Objectives

## Slide 3: Background & DBP Formation (0:45 - 1:45)
"Disinfection By-products (DBPs) are a critical issue. They form when disinfectants react with organic matter. We are specifically concerned with **Trihalomethanes (THMs)**, **Haloacetic acids (HAAs)**, and **Haloacetonitriles (HANs)** due to their health risks.
Our goal is to use Deep Learning for real-time prediction in a simulated drinking water network.
Chemically, chloramines react with organic precursors to generate nitrogenous DBPs (N-DBPs), such as Nitrosamines."

## Slide 4: Research Gap (1:45 - 2:30)
"Why do we need this research?
1.  **Dynamic Conditions**: Traditional mechanistic models struggle with the non-linear dynamics of distribution systems.
2.  **Limit of Traditional ML**: Even conventional machine learning, like Decision Trees or SVMs, often fails to capture the complex temporal dependencies inherent in our time-series data.
3.  **Monitoring Limitations**: Lab analysis is slow and expensive.
4.  **Opportunity**: We have high-frequency sensor data. Using Deep Learning enables proactive, real-time control that other methods cannot match."

## Slide 5: Objectives (2:30 - 3:00)
"Our objectives are:
1.  **Development of Deep Learning Models**: Comparing MLP, RNN, LSTM, and GRU.
2.  **Analyze Correlations (in progress)**: Investigating correlations between sensors and DBPs.
3.  **Develop Prediction Framework (in progress)**: Building a robust system for real-time monitoring."

# Section 2: Methodology & Experimental Framework

## Slide 6: System Description (3:00 - 3:30)
"We simulated a Water Distribution System with a Disinfection Tank, Retention Tank, and two Pipelines. This allows us to track water quality evolution."

## Slide 7: Data Acquisition (3:30 - 4:00)
"We use a comprehensive sensor network measuring core parameters (TRC, pH) and advanced ones (fDOM, TOC) at a 5-minute resolution."

## Slide 8: Data Preprocessing: Imputation (4:00 - 4:45)
"For preprocessing, we first handle missing values. Alternating sensors cause 30-minute gaps.
As shown in the figure, we use **Stochastic Interpolation**. This combines linear interpolation with random noise injection that matches the local variance. This approach fills the gaps while preserving the statistical properties of the signal, which is crucial for training robust models."

## Slide 9: Data Preprocessing: Time Alignment (4:45 - 5:15)
"Next is Time Alignment. Water travels **across the stages**. We correct for this hydraulic lag by shifting the input series, ensuring our model learns the transformation of the **same batch of water**."

## Slide 10: Model Architectures (5:15 - 6:00)
"We tested four architectures, visualized here:
- **MLP**: A standard feedforward network using flattened historical windows.
- **RNN**: The basic recurrent network for sequences.
- **LSTM**: Uses gating mechanisms (input, output, forget gates) to capture long-term dependencies.
- **GRU**: A streamlined variant of LSTM with update and reset gates.
These recurrent architectures are specifically chosen to handle the temporal nature of our data."

## Slide 11: Training Process (6:00 - 6:45)
"This workflow illustrates our training process.
We rely on **Mean Squared Error (MSE)** as our loss function, as it effectively penalizes large regression errors.
The **Adam optimizer** is used for its adaptive learning rate properties.
To prevent overfitting, we employ **Dropout** and **Early Stopping**—monitoring the validation loss and stopping training when it ceases to improve. The data is rigorously split into Training, Validation, and Test sets."

## Slide 12: Hyperparameter Optimization (6:45 - 7:30)
"Optimizing these models is complex. We used **Bayesian Optimization**.
This method builds a probabilistic model to intelligently select the next set of hyperparameters to evaluate, balancing exploration and exploitation. It finds optimal configurations much faster than grid search."

## Slide 13: Algorithm Optimizing (Rate of Change) (7:30 - 8:15)
"We optimized our algorithms with two strategies.
First, the **Normalized Rate of Change**.
Absolute concentrations are dominated by the stable upstream value. The chemical transformation causes only small deviations. By predicting the 'Rate'—the relative change—we force the model to focus on the *kinetics* of the reaction occurring within the pipe, rather than just memorizing the input level."

## Slide 14: Algorithm Optimizing (Decoupled Model) (8:15 - 9:00)
"Second, **Decoupled Modeling**.
Total Residual Chlorine (TRC) behaves very differently from stable parameters like pH or TOC; it decays rapidly. In a combined model, its accuracy often suffers.
We solved this by building a dedicated **TRC Model** exclusively for chlorine, and a separate **Other Parameter Model** for the rest. This allows specialized tuning for the most critical parameter."

# Section 3: Results & Conclusion

## Slide 15: Performance Metrics (9:00 - 9:45)
"Our results show that:
- For TRC, **LSTM with Rate regression** is effective. `Rate` helps capture the decay kinetics.
- For stable parameters (pH, TOC), **GRU with Value regression** is superior. `Value` works best here as absolute concentrations are stable.
(Note: MLPHIS was excluded as standard MLP/RNN/LSTM/GRU provided sufficient insight)."

## Slide 16: Prediction Visualization (9:45 - 10:15)
"The plots confirm that our best models closely track the actual sensor trends."

## Slide 17: Conclusion & Future Work (10:15 - 11:15)
"Key Findings:
1.  **General Fit**: Neural Networks are highly capable of fitting this complex water quality data.
2.  **Performance**: GRU and LSTM are superior, validating the importance of temporal modeling.
3.  **Strategy**: The choice of regression target is key—'Rate' for reactive parameters like TRC, 'Value' for stable ones.

Future Work:
We will focus on a **Second-Stage Model** to map these predicted water quality references (TRC, pH, TOC) to actual DBP concentrations (THMs, HAAs)."

## Slide 18: Q&A (11:15 - 15:00)
"Thank you. Questions?"
