#These are the libraries I loaded in to run the modelling procedures
library(tidyverse)
library(tidymodels)
library(vroom)
library(stacks)
library(themis)
library(bonsai)
library(lightgbm)
library(discrim)
library(themis)

#Loading in the training and test data from the provided datasets
otto_tr <- vroom("./Otto-Product-Classification/train.csv")
otto_te <- vroom("./Otto-Product-Classification/test.csv")

##EDA
#There doesn't appear to be any missing data (this is good)
plot_missing(otto_tr)

#Classes 2 and 6 are by far the most dominant within this model. There might
#Be some data imbalancing here, but smote would be expensive in a model this massive
#I'll run a normal model and then see if it is necessary (SMOTE didn't work well)
otto_tr %>%
  group_by(target) %>%
  summarize(n = n())

glimpse(otto_tr)

#Removing id from the training data (not a feature)
otto_tr <- otto_tr %>%
  select(-id)

#This is the recipe that I tinkered with. For Naive Bayes, normalization and pca
#Provided small improvements. SMOTE did not, so I opted not to include it in the
#recipe
my_recipe <- recipe(target ~ ., data = otto_tr) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .99)
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = otto_tr)

##Naive Bayes

#Naive Bayes model (the Laplace and smoothness parameter will be tuned through CV)
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

#Workflow for naive bayes (standard recipe)
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

#Tuning grid with only 9 total options to keep computation time low
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 3)

#Using 5-fold cross validation
folds <- vfold_cv(otto_tr, v = 5, repeats = 1)

#Actual CV workflow. This will record the log loss of each possible combination
#of Laplace and smoothness
CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mn_log_loss))

#Selects the best combination of parameters
bestTune <- CV_results %>%
  select_best("mn_log_loss")

#Apply the bestTune to the original workflow
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = otto_tr)

#Predict the test dataset using the finalized model
#Format the predictions into the valid format and write it off using a vroom function
#Commented out because the final model is XGBoost
# otto_predictions <- forest_workflow %>% predict(new_data = otto_te, type = "prob") %>%
#   bind_cols(otto_te$id) %>%
#   rename("id" = "...10",
#          "Class_1"= ".pred_Class_1",
#          "Class_2"= ".pred_Class_2",
#          "Class_3"= ".pred_Class_3",
#          "Class_4"= ".pred_Class_4",
#          "Class_5"= ".pred_Class_5",
#          "Class_6"= ".pred_Class_6",
#          "Class_7"= ".pred_Class_7",
#          "Class_8"= ".pred_Class_8",
#          "Class_9"= ".pred_Class_9") %>%
#   select(id, everything())
# 
# vroom_write(otto_predictions, "./Otto-Product-Classification/submission_nb.csv", col_names = TRUE, delim = ", ")


####Random forest workflow

#Random forest model. The tuning parameters were selected by hand rather than by CV
#becase of the exceptionally long computation time
tree_mod <- rand_forest(mtry = 10,
                        min_n = 50,
                        trees = 750) %>%
  set_engine("ranger") %>%
  set_mode("classification")


##Workflow (standard recipe)
forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_mod) %>%
  fit(data = otto_tr)

#Prior CV process was implemented, but computations proved to be a little too long
# tuning_grid <- grid_regular(mtry(range = c(1, 50)),
#                             min_n(),
#                             levels = 3)
# 
# folds <- vfold_cv(otto_tr, v = 5, repeats = 1)
# 
# CV_results <- forest_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(mn_log_loss))
# 
# bestTune <- CV_results %>%
#   select_best("mn_log_loss")
# 
# final_wf <- forest_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = otto_tr)

#Predict the test dataset using the finalized model
#Format the predictions into the valid format and write it off using a vroom function
#Commented out because the final model is XGBoost
# otto_predictions <- forest_workflow %>% predict(new_data = otto_te, type = "prob") %>%
#   bind_cols(otto_te$id) %>%
#   rename("id" = "...10",
#          "Class_1"= ".pred_Class_1",
#          "Class_2"= ".pred_Class_2",
#          "Class_3"= ".pred_Class_3",
#          "Class_4"= ".pred_Class_4",
#          "Class_5"= ".pred_Class_5",
#          "Class_6"= ".pred_Class_6",
#          "Class_7"= ".pred_Class_7",
#          "Class_8"= ".pred_Class_8",
#          "Class_9"= ".pred_Class_9") %>%
#   select(id, everything())
# 
# vroom_write(otto_predictions, "./Otto-Product-Classification/submission_nb.csv", col_names = TRUE, delim = ", ")


##XGBoosting


#XGboosting operates the best without any additional modifications
boost_recipe <- recipe(target ~ ., data = otto_tr)
prep_boost <- prep(boost_recipe)
baked_tr <- bake(prep_boost, new_data = otto_tr)


#Xgboosting ended up being the most effective model, and faster to compute
#than random forests. Shoutout to Datacanary's work for the tuning parameters used
boost_model <- boost_tree(learn_rate = .3,
                          trees = 50,
                          tree_depth = 6) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

#This appies and fits the xgboosting model
boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model) %>%
  fit(data = otto_tr)

#Creates predictions and formats them for submission
otto_predictions <- boost_wf %>% predict(new_data = otto_te, type = "prob") %>%
  bind_cols(otto_te$id) %>%
  rename("id" = "...10",
         "Class_1"= ".pred_Class_1",
         "Class_2"= ".pred_Class_2",
         "Class_3"= ".pred_Class_3",
         "Class_4"= ".pred_Class_4",
         "Class_5"= ".pred_Class_5",
         "Class_6"= ".pred_Class_6",
         "Class_7"= ".pred_Class_7",
         "Class_8"= ".pred_Class_8",
         "Class_9"= ".pred_Class_9") %>%
  select(id, everything())

vroom_write(otto_predictions, "./Otto-Product-Classification/submission.csv", col_names = TRUE, delim = ", ")