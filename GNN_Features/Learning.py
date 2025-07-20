from agent_GNN_Features import Agent_GNN_Features

agent = Agent_GNN_Features(0.01, 0.2)

agent.train_net("../Data/mixed")

agent.save_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")

agent.save_learning_curve("/home/mi/maxa55/Master_Thesis/Results/Learning Curves/gnn_features/lc_swap")

#agent.plot_learning_curve("learning_curve_noisy.png")

