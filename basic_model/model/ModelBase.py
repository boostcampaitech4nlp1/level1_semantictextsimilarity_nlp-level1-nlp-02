import torch
import torch.nn as nn

###################################
## Model for Electra based model ##
###################################

class Base(nn.Module):
    def __init__(self, transformer, first_in):
        super(Base, self).__init__()

        ## Dimension
        self.first_in = first_in
        self.h_dim = 512
        self.classification_dim = 1
        self.score_dim = 1

        ## Transformer model
        self.transformer = transformer

        ## ReLU & Sigmoid
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        ## Regression layers
        self.layer1 = nn.Linear(self.first_in, self.h_dim)
        self.layer2 = nn.Linear(self.h_dim, self.h_dim)
        self.layer_out = nn.Linear(self.h_dim, self.score_dim)

        ## Classification layers
        self.layer1_1 = nn.Linear(self.first_in, self.h_dim)
        self.layer2_1 = nn.Linear(self.h_dim, self.h_dim)
        self.layer_out_1 = nn.Linear(self.h_dim, self.classification_dim)
    
    def forward(self, idz, attentions, token_types):
        x = self.transformer(
            idz,
            token_type_ids = token_types,
            attention_mask = attentions,
            return_dict = False,
        )[0][:,0,:]

        ## Regression layer
        x_regression = self.relu(self.layer1(x))
        x_regression = self.relu(self.layer2(x_regression))
        score = self.layer_out(x_regression)

        ## Classification layer
        x_classification = self.relu(self.layer1_1(x))
        x_classification = self.relu(self.layer2_1(x_classification))
        bi_class = self.sigmoid(self.layer_out_1(x_classification))

        return score, bi_class