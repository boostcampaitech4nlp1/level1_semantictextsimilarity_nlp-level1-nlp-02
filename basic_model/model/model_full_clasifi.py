import torch
import torch.nn as nn

###################################
## Model for Electra based model ##
###################################

class FullClasifi(nn.Module):
    def __init__(self, transformer, first_in):
        super(FullClasifi, self).__init__()

        ## Dimension
        self.first_in = 768
        self.h_dim = 512
        self.multi_classification_dim = 31
        self.bi_classification_dim = 1

        ## Transformer model
        self.transformer = transformer

        ## ReLU & Sigmoid
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        ## Multi-Classification layers
        self.layer1 = nn.Linear(self.first_in, self.h_dim)
        self.layer2 = nn.Linear(self.h_dim, self.h_dim)
        self.layer_out = nn.Linear(self.h_dim, self.multi_classification_dim)

        ## Bi-Classification layers
        self.layer1_1 = nn.Linear(self.first_in, self.h_dim)
        self.layer2_1 = nn.Linear(self.h_dim, self.h_dim)
        self.layer_out_1 = nn.Linear(self.h_dim, self.bi_classification_dim)
    
    def forward(self, idz, attentions, token_types):
        x = self.transformer(
            idz,
            token_type_ids = token_types,
            attention_mask = attentions,
            return_dict = False,
        )[0][:,0,:]

        ## Multi-Classification layer
        x_multiclass = self.relu(self.layer1(x))
        x_multiclass = self.relu(self.layer2(x_multiclass))
        multi_class = self.layer_out(x_multiclass)

        ## Bi-Classification layer
        x_classification = self.relu(self.layer1_1(x))
        x_classification = self.relu(self.layer2_1(x_classification))
        bi_class = self.sigmoid(self.layer_out_1(x_classification))

        return multi_class, bi_class