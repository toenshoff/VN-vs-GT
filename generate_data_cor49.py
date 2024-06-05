import os
import networkx as nx
import pickle
import numpy as np


def generate_train_data(l_range, r_range, num_graphs_train, num_graphs_val, mean_degree):
    
    train_val = {}
    for i, num_graphs in enumerate([num_graphs_train, num_graphs_val]):
        # sample l and r
        l_s = np.random.randint(low=l_range[0], high=l_range[1]+1, size=(num_graphs,))
        r_s = np.random.randint(low=r_range[0], high=r_range[1]+1, size=(num_graphs,))
        w_num_node_s = l_s * r_s
        node_count_s = l_s + w_num_node_s
        
        # generate graphs
        mean_degree_s = np.random.randint(low=mean_degree[0], high=mean_degree[1]+1, size=(num_graphs,))
        edge_count_s = ((node_count_s/2) * mean_degree_s)//1
        graph_list = [nx.gnm_random_graph(node_count_s[i], edge_count_s[i]) for i in range(num_graphs)]
        
        curr_graphs = [graph_list, l_s, r_s, w_num_node_s]
        train_val[i] = curr_graphs
        
    dataset_dict = {"train": train_val[0], "val": train_val[1], "test": train_val[1]}
    
    os.makedirs(os.path.join(dataset_dir, 'Cor49Train', 'raw'))

    for name in ['train', 'val', 'test']:
        file = open(os.path.join(dataset_dir, 'Cor49Train', 'raw', f'{name}.pickle'), 'wb')
        pickle.dump(dataset_dict[name], file)
        file.close()
        
        
def generate_test_data(l_list, r_list, num_graphs_test, mean_degree):
    dummy_graph = [nx.gnm_random_graph(3, 3)]
    for l in l_list:
        for r in r_list:
            raw_dir_name = os.path.join(dataset_dir, f'Cor49-{l}-{r}', 'raw')
            os.makedirs(raw_dir_name)
            
            # generate graphs
            w_num_node = l*r
            node_count = l+ w_num_node
            
            curr_mean_degree = np.random.randint(low=mean_degree[0], high= mean_degree[1]+1, size=(num_graphs_test,))
            train_edge_count = ((node_count/2) * curr_mean_degree)//1
            
            graph_list = [nx.gnm_random_graph(node_count, train_edge_count[i]) for i in range(num_graphs_test)]
            
            test_graphs = [graph_list, np.full((num_graphs_test,), l), np.full((num_graphs_test,), r), np.full((num_graphs_test,), w_num_node)]
            dataset_dict = {"train": [dummy_graph, [1], [2], [2]], "val": [dummy_graph, [1], [2], [2]], "test": test_graphs}
            
            for name in ['train', 'val', 'test']:
                file = open(os.path.join(raw_dir_name, f'{name}.pickle'), 'wb')
                pickle.dump(dataset_dict[name], file)
                file.close()


if __name__ =="__main__":
    np.random.seed(0)
    dataset_dir = './datasets'
    num_graphs_train = 100000
    num_graphs_val = 100
    
    train_l_range = [1, 10]
    train_r_range = [1, 5]
    mean_degree = [1, 5]

    num_graphs_test = 100
    test_l = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    test_r = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # create train dataset
    generate_train_data(train_l_range, train_r_range, num_graphs_train, num_graphs_val, mean_degree)
    generate_test_data(test_l, test_r, num_graphs_test, mean_degree)
