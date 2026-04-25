# Final Presentation Script (20 Minutes)

*Note: This script provides the spoken narrative for each slide. The presentation aims for a total duration of 20 minutes, allocating roughly 45 seconds per slide, and incorporating a 1-minute 30-second live video demonstration. The language is conversational and easy to follow.*

---

## Slide 1: Title Slide (0:00 - 0:30)
"Hi everyone. I'm Zhou Dafu. Today I'm going to share our final report on using Machine Learning to predict Disinfection By-products. I'd like to thank Professor Hu Jiangyong for supervising this project, and thank you all for coming."

## Slide 2: Outline (0:30 - 1:00)
"Here's a quick look at what we'll cover today. First, I'll talk about the background of the problem and what we wanted to achieve. Then, I'll explain how we set up our data and the different AI models we tested. After that, I'll show you a software app we built to make these models easy to use. Finally, we'll go through the results and talk about what comes next."

---
# Section 1: Introduction & Objectives

## Slide 3: Background & DBP Formation (1:00 - 1:50)
"Let's start with why this matters. Adding chlorine makes our drinking water safe, but it also reacts with organic stuff in the water to create Disinfection By-products, or DBPs. Some of these are really bad for our health. To track this, we look at the chemical reactions—specifically how chloramines form and react with nitrogen in the water. Our main goal is to use simple sensor data, like pH and Chlorine levels, to predict these harmful DBPs in real-time."

## Slide 4: Limitations of Traditional Approaches (1:50 - 2:30)
"So why use Deep Learning? Well, traditional physical models have a hard time dealing with how fast and unpredictably water conditions change in real pipes. Plus, checking water in a lab is slow and very expensive. Deep Learning is great here because it can look at high-frequency sensor data and automatically find the hidden patterns over time, without us having to figure out all the math manually."

## Slide 5: Transfer Learning & Fine-Tuning (2:30 - 3:15)
"One big challenge we tackled is Transfer Learning. You see, water quality changes a lot depending on the source and the temperature. Training a brand-new model for every single change takes too much time and computer power. So, we use a Fine-Tuning strategy. We take a pre-trained model and just adapt it to the new conditions. We tested three ways to do this: Full Fine-Tuning, Partial Fine-Tuning, and using small Adapter layers."

## Slide 6: Objectives (3:15 - 4:00)
"To sum it up, our goals for this final stage were: First, to test out several models—from fast tree-based models like XGBoost to advanced neural networks like LSTMs and Transformers. Second, to see how well the Transformer can adapt to new temperatures and water sources. And third, to wrap all this up into a working software interface for real operators to use."

---
# Section 2: Methodology

## Slide 7: System Description (4:00 - 4:30)
"Moving on to how we did this. We built a simulated Water Distribution System with a Disinfection Tank, a Retention Tank, and two pipelines. We placed sensors throughout the system to track the water quality every 5 minutes."

## Slide 8: Data Preprocessing: Stochastic Imputation (4:30 - 5:15)
"But real data is messy. We had these annoying 30-minute gaps in our sensor data because of how the sensors switched back and forth. To fix this, we used something called Stochastic Imputation. Basically, we filled the gaps using local trends but also added in realistic random noise. This way, the filled data still looks and acts like real sensor readings."

## Slide 9: Data Preprocessing: Data Segmentation (5:15 - 5:45)
"Next, we had to prepare the data for the models. Instead of manually trying to match up timestamps as water flows down the pipe, we used a sliding window approach. This lets the models figure out the delays naturally. We also made sure to split our data chronologically into training, validation, and testing sets so our results are trustworthy."

## Slide 10: Model Architectures Overview (5:45 - 6:30)
"For the models themselves, we looked at two main camps. The first camp is GBDTs—like XGBoost and LightGBM. They are very fast but they mostly predict one thing at a time. The second camp is Neural Networks, which are built for predicting multiple things at once. We tested basics like MLP, recurrent networks like LSTM and GRU, and state-of-the-art sequence models like the Transformer and Mamba."

## Slide 11: Transformer for Time Series: Why Encoder-Only? (6:30 - 7:15)
"I want to briefly focus on the Transformer. It uses an 'attention' mechanism to look at the whole sequence of data all at once, which is incredibly powerful. Normally, Transformers have two parts: an Encoder and a Decoder. But since we are just trying to predict numbers from a fixed set of history, we threw out the Decoder. This 'Encoder-Only' setup saves a lot of memory, prevents overfitting, and keeps all the predictive power we need."

## Slide 12: Model Training Configuration (7:15 - 7:45)
"Our training setup was standard to keep things fair. We used Mean Squared Error to measure mistakes and the Adam optimizer to learn. We also made sure to fine-tune the settings for every single model using Bayesian Optimization."

## Slide 13: Model Training Workflow (7:45 - 8:15)
"Here is what our overall training workflow looks like. It shows the whole pipeline: from raw data coming in, going through cleaning and tuning, to training the models and finally testing them on unseen data."

## Slide 14: Fine-Tuning Workflow (8:15 - 8:45)
"For our transfer learning tests, we set up two baseline Transformer models at $29^{\circ}\text{C}$ for our two different water sources. Then, we applied our fine-tuning strategies to adapt them to $35^{\circ}\text{C}$ data. This let us see exactly how well the models could learn the new chemical reactions driven purely by the heat."

---
# Section 3: GUI Development

## Slide 15: Frontend Software Application (8:45 - 9:30)
"We didn't just want this to stay an academic project, so we built a real Graphical User Interface. For the tech folks, we used Vite, React, and Electron. This gives us a very fast, clean desktop app that can run our heavy Python machine learning code right on the operator's computer."

## Slide 16: Live Demonstration (Video) (9:30 - 11:00)
"Now, I'm going to show a quick 1-and-a-half-minute video of the software in action. You'll see how easy it is to load the data, train a model, and get real-time warnings."
*(Play Video - Pause Presentation)*

---
# Section 4: Results & Conclusion

## Slide 17: Evaluation Environment & GBDT Metrics (11:00 - 11:45)
"Alright, let's talk about the results. Everything was run on a strong PC with an RTX 4070 graphics card. Starting with the GBDT models, XGBoost was the clear winner here, getting the lowest error. But we have to remember: GBDTs don't really understand the concept of time naturally, and they can't predict values higher or lower than what they've seen in training."

## Slide 18: GBDT Models: Visualization (11:45 - 12:30)
"You can see this clearly in these plots. The top row shows the predictions tracking the actual values, and the bottom row shows the scatter plots. XGBoost on the left tracks the real data pretty tightly, but LightGBM and CatBoost are quite messy and noisy."

## Slide 19: Neural Network Metrics (12:30 - 13:30)
"When we look at the Neural Networks, the story changes. Here, the LSTM actually got the best score overall, beating even the Transformer and Mamba. This makes sense because LSTM gates are specifically designed to handle this kind of medium-length number sequence very well. The Transformer is great, but its true power usually shows up in much larger, messier datasets."

## Slide 20: Neural Networks: Visualization (Part 1) (13:30 - 14:15)
"Looking at the visual predictions for the simpler neural networks—like MLP, RNN, and GRU—they all do a really solid job. The GRU on the right is almost as good as the LSTM but uses fewer resources."

## Slide 21: Neural Networks: Visualization (Part 2) (14:15 - 15:00)
"And here we have the LSTM, the Transformer, and Mamba. The LSTM on the left has the smoothest and tightest fit. The Transformer and Mamba also capture the main trends, but they bounce around a bit more locally. So for this specific task, classic recurrent models still have a slight edge in pure accuracy."

## Slide 22: Transformer Fine-Tuning Performance (15:00 - 16:30)
"Finally, let's look at how Transfer Learning went. This table shows the errors when moving from $29^{\circ}\text{C}$ to $35^{\circ}\text{C}$. For the cleaner CAWW water source, Full Fine-Tuning—which means updating everything—worked the best. But for the noisier LSWW water source, Partial Fine-Tuning was the winner. By freezing the core parts of the model, we stopped it from getting confused by the extra noise. Sadly, the Adapter approach didn't work well for this specific numerical task."

## Slide 23: Conclusion (16:30 - 17:30)
"So, what are our main takeaways? First, the LSTM is still the most accurate baseline model for this kind of water data. Second, Transformer models are amazing at adapting to new situations—like changes in temperature or water source—if you use Partial Fine-Tuning. And third, we proved this can all be packaged into a useful software tool for the industry."

## Slide 24: Future Work (17:30 - 18:00)
"Looking ahead, we want to do two things. First, we want to add 'Model Interpretability' using SHAP values. This will tell us exactly which sensors are causing the model to make its predictions. Second, we want to build a second-stage model that takes our current predictions and maps them directly to the actual, lab-tested DBP concentrations."

## Slide 25: Conceptual Roadmap (18:00 - 18:45)
"This slide just gives a quick visual idea of that roadmap, showing how our current sensor predictions will plug into that future physical mapping."

## Slide 26: Q&A (18:45 - 20:00)
"And that brings me to the end of my talk. Thank you all so much for listening. I'd be happy to answer any questions you have."
