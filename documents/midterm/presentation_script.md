# Midterm Presentation Script (Consolidated - 15 Minutes)

## Slide 1: Title Slide (0:00 - 0:30)
"Good morning/afternoon everyone. My name is Zhou Dafu. Today I will present my midterm progress on applying Supervised Learning methods to understand DBP formation mechanisms in water distribution systems. This work is supervised by Prof. Hu Jiangyong and mentored by Mr. Sun Yuanpeng."

## Slide 2: Outline (0:30 - 0:45)
"My presentation is divided into three main sections:
1.  Introduction and Objectives.
2.  Methodology and Experimental Framework.
3.  Results and Conclusion."

# Section 1: Introduction & Objectives

## Slide 3: Background & DBP Formation (0:45 - 2:30)
"Let's start with the background. Disinfection By-products, or DBPs, are a major health concern in water treatment. They form when disinfectants react with organic matter, and many are carcinogenic. Our goal is to use Deep Learning to predict these formation events in real-time.
Chemically, in our alkaline chloramination system, the key reaction starts with ammonia and hypochlorite forming monochloramine. These chloramines then react with organic precursors to form intermediates, which eventually oxidize into harmful N-DBPs like Nitrosamines."

## Slide 4: Objectives (2:30 - 3:30)
"Our study has three primary objectives:
1.  To evaluate and compare different Deep Learning architectures (MLP, RNN, LSTM, GRU).
2.  To analyze the correlations between standard sensor data (like TRC and pH) and DBP formation.
3.  To develop a robust prediction framework that can operate in real-time."

# Section 2: Methodology & Experimental Framework

## Slide 5: System Description (3:30 - 4:15)
"Moving on to our methodology. We simulated a Water Distribution System comprising a Disinfection Tank, a Retention Tank for storage, and two pipelines. This setup allows us to monitor the chemical evolution of water as it travels through the network."

## Slide 6: Data Acquisition (4:15 - 5:00)
"We collect high-frequency sensor data at 5-minute intervals.
Our sensor network covers all key stages. We measure core parameters like TRC, pH, and Conductivity, as well as advanced parameters like fDOM and TOC. This comprehensive dataset captures the water quality profile throughout the system."

## Slide 7: Data Preprocessing: Imputation (5:00 - 5:30)
"Raw data requires careful preprocessing. The first challenge is missing values due to alternating sensors.
Standard mean imputation would reduce the variance and distort the signal.
Instead, we used a stochastic interpolation method. We first linearly interpolate between known points and then inject random noise that matches the local variance. This preserves the statistical properties of the water quality dynamics."

## Slide 8: Imputation (Visual) (5:30 - 5:45)
"This figure demonstrates our imputation method. The red points represent the imputed values filling the gaps, maintaining the natural fluctuation of the sensor readings."

## Slide 9: Data Preprocessing: Time Alignment (5:45 - 6:15)
"The second challenge is hydraulic delay. Water takes time to travel from the tanks to the pipelines.
If we directly correlate timestamps, we get incorrect mappings.
We solved this by calculating the residence time and shifting the input data series forward. This ensures the model learns the transformation of the *same* parcel of water."

## Slide 10: Time Alignment (Visual) (6:15 - 6:30)
"This diagram illustrates the alignment process. By shifting the input series (blue) to match the output (orange), we align the peaks and troughs, allowing the model to learn the correct causal relationship."

## Slide 11: Model Architectures (6:30 - 7:15)
"We explored four neural network architectures:
**MLP**: A simple feedforward network as a baseline.
**RNN**: Capable of handling time-series but prone to vanishing gradients.
**LSTM and GRU**: Advanced recurrent networks with gating mechanisms. These are specifically designed to capture long-term dependencies, which is crucial for modeling the slow chemical kinetics in our system."

## Slide 12: Training Process (7:15 - 7:45)
"Our training process is rigorous.
We use **Mean Squared Error (MSE)** as the loss function because it penalizes larger errors, which is ideal for regression.
We use the **Adam optimizer** for fast convergence.
To prevent overfitting, we employ **Dropout** and **Early Stopping**."

## Slide 13: Training Process (Workflow) (7:45 - 8:00)
"Here is the complete workflow. We split the data into training, validation, and test sets. The model is trained on the training set, tuned on the validation set, and finally evaluated on the unseen test set to ensure generalizability."

## Slide 14: Hyperparameter Optimization (8:00 - 8:45)
"Optimizing these models is complex. Instead of manual tuning, we used **Bayesian Optimization**.
This method builds a probabilistic model of the objective function to intelligently select the next set of hyperparameters to evaluate. It balances exploration of new regions and exploitation of known good regions, finding optimal configurations much faster than grid search."

## Slide 15: Strategy 1: Normalized Rate of Change (8:45 - 9:30)
"To further improve performance, we devised two specific strategies.
First is the **Normalized Rate of Change Regression**.
Instead of predicting absolute values, we predict the relative change.
Since absolute concentrations are dominated by the stable upstream value, predicting the 'Rate' forces the model to focus on the *kinetics* of the reaction occurring within the pipe."

## Slide 16: Strategy 2: Decoupled TRC Model (9:30 - 10:15)
"Second is the **Decoupled TRC Model**.
Total Residual Chlorine (TRC) is the primary disinfectant and decays rapidly, behaving very differently from stable parameters.
In a combined model, its accuracy suffers.
So, we built a dedicated model exclusively for TRC, allowing for specialized tuning, while a second model handles all other parameters."

# Section 3: Results & Conclusion

## Slide 17: Performance Metrics (10:15 - 11:30)
"Here are our results for Pipeline 2.
We found that the 'Rate' regression strategy (with LSTM) was most effective for predicting TRC.
However, for stable parameters like pH and TOC, the direct 'Value' regression (with GRU) performed best. This highlights that different water quality parameters require different modeling approaches."

## Slide 18: Prediction Visualization (11:30 - 12:30)
"This plot shows our model predictions versus the actual sensor data.
As you can see, the models track the trends very closely for TRC, pH, and TOC, validating the effectiveness of our approach."

## Slide 19: Conclusion & Future Work (12:30 - 13:30)
"In conclusion, Deep Learning models, particularly GRU and LSTM, are highly effective for this task. The choice of regression strategy—Rate vs. Value—is critical depending on the parameter.
For future work, we plan to:
1.  Test advanced Transformer models like PatchTST.
2.  Focus on **Interpretability** using tools like SHAP to understand the 'why' behind predictions.
3.  Build a second-stage model to predict actual DBP concentrations."

## Slide 20: Q&A (13:30 - 15:00)
"Thank you. I am now open to any questions."
