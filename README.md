# Track-source-of-water-contamination-using-machine-learning

Current microbial source tracking techniques that rely on grab samples analyzed by individual endpoint assays are inadequate to explain microbial sources across space and time. Modeling and predicting host sources of microbial contamination could add a useful tool for watershed management. 

In this study, we tested and evaluated machine learning models to predict the major sources of microbial contamination in water. 

We examined the relationship between microbial sources, land cover, weather, and hydrologic variables in a watershed in Northern California, USA. 

Six models, including K-nearest neighbors (KNN), Naïve Bayes, Support vector machine (SVM), simple neural network (NN), Random Forest and XGBoost, were built to predict major microbial sources using land cover, weather and hydrologic variables. 

The results showed that these models successfully predicted microbial sources classified into two categories (human and non-human), with the average accuracy ranging from 69% (Naïve Bayes) to 88% (XGBoost). The area under curve (AUC) of the receiver operating characteristic (ROC) illustrated XGBoost had the best performance (average AUC=0.88), followed by Random Forest (average AUC=0.84), and KNN (average AUC=0.78). 

The importance index obtained from Random Forest indicated that precipitation and temperature were the two most important factors to predict the dominant microbial source. 

These results suggest that machine learning models, particularly, XGBoost, can predict the dominant sources of microbial contamination based on their relationship with daily weather and land cover, providing a powerful tool to understand microbial sources in water.


Key words: Fecal contamination, microbial source tracking, land use, rainfall, machine learning, XGBoost
