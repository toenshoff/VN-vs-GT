import os
import networkx as nx
import pickle
import numpy as np


def generate_train_data(train_n_nodes, mean_degree, num_graphs_train, num_graphs_val):
    train_node_count = np.random.randint(low=train_n_nodes[0], high=train_n_nodes[1], size=(num_graphs_train,))
    train_mean_degree = np.random.randint(low=mean_degree[0], high=mean_degree[1], size=(num_graphs_train,))
    assert train_node_count.shape[0] == train_mean_degree.shape[0] == num_graphs_train
    train_edge_count = ((train_node_count/2) * train_mean_degree)//1
    train_graph_list = [nx.gnm_random_graph(train_node_count[i], train_edge_count[i]) for i in range(num_graphs_train)]
    
    val_node_count = np.random.randint(low=train_n_nodes[0], high=train_n_nodes[1], size=(num_graphs_val,))
    val_mean_degree = np.random.randint(low=mean_degree[0], high=mean_degree[1], size=(num_graphs_val,))
    val_edge_count = ((val_node_count/2) * val_mean_degree)//1
    val_graph_list = [nx.gnm_random_graph(val_node_count[i], val_edge_count[i]) for i in range(num_graphs_val)]
    
    dataset_dict = {"train": train_graph_list, "val": val_graph_list, "test": val_graph_list}
    
    os.makedirs(os.path.join(dataset_dir, 'Cor44Train', 'raw'))

    for name in ['train', 'val', 'test']:
        file = open(os.path.join(dataset_dir, 'Cor44Train', 'raw', f'{name}.pickle'), 'wb')
        pickle.dump(dataset_dict[name], file)
        file.close()


def generate_test_data(test_n_nodes, mean_degree, num_graphs_test):
    dummy_graph = [nx.gnm_random_graph(3, 3)]
    for node_count in test_n_nodes:
        raw_dir_name = os.path.join(dataset_dir, f'Cor44-{node_count}', 'raw')
        os.makedirs(raw_dir_name)
        test_mean_degree = np.random.randint(low=mean_degree[0], high=mean_degree[1], size=(num_graphs_test,))
        edge_count = ((node_count/2) * test_mean_degree)//1
        assert edge_count.shape[0] == num_graphs_test
        graph_list = [nx.gnm_random_graph(node_count, edge_count[i]) for i in range(num_graphs_test)]
        
        dataset_dict = {"train": dummy_graph, "val": dummy_graph, "test": graph_list}
        
        for name in ['train', 'val', 'test']:
            file = open(os.path.join(raw_dir_name, f'{name}.pickle'), 'wb')
            pickle.dump(dataset_dict[name], file)
            file.close()


if __name__ == "__main__":
    np.random.seed(0)
    dataset_dir = './datasets'

    num_graphs_t = 100000
    num_graphs_v = 100
    num_graphs_test = 100

    train_num_nodes = [10, 50] # uses it also for validation
    test_num_nodes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    mean_degree = [2, 5]

    generate_train_data(train_num_nodes, mean_degree, num_graphs_t, num_graphs_v)
    generate_test_data(test_num_nodes, mean_degree, num_graphs_test)
