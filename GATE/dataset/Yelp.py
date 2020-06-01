from dataset.Dataset import DataSet


# Amazon review dataset
class ON(DataSet):
    def __init__(self):
        self.dir_path = './dataset/data/yelp/ON/'
        self.user_record_file = 'user_records.pkl'
        self.user_mapping_file = 'user_mapping.pkl'
        self.item_mapping_file = 'item_mapping.pkl'
        self.item_content_file = 'word_counts.txt'
        self.item_relation_file = 'item_relation.pkl'
        self.item_word_seq_file = 'review_word_sequence.pkl'
        
        # data structures used in the model
        self.num_users = 16983  
        self.num_items = 12342
        self.vocab_size = 7450

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        word_seq = self.load_pickle(self.dir_path + self.item_word_seq_file)

        self.num_users = len(user_mapping)
        self.num_items = len(item_mapping)

        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)

        train_set, test_set = self.split_data_randomly(inner_data_records, seed)

        train_matrix = self.generate_rating_matrix(train_set, self.num_users, self.num_items)
        item_content_matrix = self.load_item_content(self.dir_path + self.item_content_file, self.vocab_size)
        item_relation_matrix = self.load_pickle(self.dir_path + self.item_relation_file)

        return train_matrix, train_set, test_set, item_content_matrix, item_relation_matrix, word_seq