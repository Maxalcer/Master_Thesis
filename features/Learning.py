from agent_features_fixed import Agent_Features_Fixed

agent = Agent_Features_Fixed(0.01, 0.2)

agent.train_net("../Data/mixed")

agent.save_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_long.py")

agent.save_learning_curve("/home/mi/maxa55/Master_Thesis/Results/Learning Curves/feature/fixed_mix_long")

#agent.plot_learning_curve("learning_curve_noisy.png")

