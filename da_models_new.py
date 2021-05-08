import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.optim as optim
from word_models import EventExtractor

# Gradient-reversal layer
class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -1 * ctx.constant * grad_output, None


gradreverse = GradReverse.apply


# Adversarial classifier
class Adversary(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(Adversary, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = 2*n_layers-1

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.append(nn.ReLU())
        for i in range(1, n_layers-1):
            self.layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, features):
        for i in range(self.n_layers):
            features = self.layers[i](features)
        return features


# Complete adversarial framework
class AdversarialEventExtractor(nn.Module):

    def __init__(self, vocab_size, emb_size, rep_hidden_size, num_classes, adv_hidden_size, adv_layers, num_domains, adv_coeff, dropout, is_bidir, rep_learner='word', pos_vocab_size=None):
        super(AdversarialEventExtractor, self).__init__()
        self.event_extractor = EventExtractor(vocab_size, emb_size, rep_hidden_size, num_classes, dropout, is_bidir, rep_learner, pos_vocab_size)
        self.pooling_layer = nn.AdaptiveMaxPool1d(1)
        self.adv_classifier = Adversary(self.event_extractor.rep_size, adv_hidden_size, num_domains, adv_layers)
        self.rep_type = rep_learner
        self.adv_coeff = adv_coeff

    def forward(self, event_data, domain_data):

        # Forward pass for loss 1: Only adversarial classifier weights
        domain_outputs, event_outputs, event_domains = [], [], []
        if self.rep_type == "word" or self.rep_type.startswith("bert"):
            sents, labels, lengths, masks = domain_data
            domain_reps = self.event_extractor.rep_learner(sents, lengths, masks)
            pooled_domain_rep = self.pooling_layer(domain_reps.permute(0,2,1))
            domain_outputs = self.adv_classifier(pooled_domain_rep.squeeze(-1))

            # Forward pass for loss 2: Representation learner + Event classifier weights
            # This pass uses the gradient reversal layer just before the adversarial classifier
            sents, labels, lengths, masks = event_data
            event_outputs = self.event_extractor(sents, lengths, masks)  # Remove event_reps after dumping BERT
            pooled_event_rep = self.pooling_layer(self.event_extractor.reps.permute(0,2,1))
            event_domains = self.adv_classifier(gradreverse(pooled_event_rep.squeeze(-1), self.adv_coeff))

        elif self.rep_type == "pos":
            sents, pos, labels, lengths, masks = domain_data
            domain_reps = self.event_extractor.rep_learner(sents, pos, lengths, masks)
            pooled_domain_rep = self.pooling_layer(domain_reps.permute(0,2,1))
            domain_outputs = self.adv_classifier(pooled_domain_rep.squeeze(-1))

            # Forward pass for loss 2: Representation learner + Event classifier weights
            # This pass uses the gradient reversal layer just before the adversarial classifier
            sents, pos, labels, lengths, masks = event_data
            event_reps = self.event_extractor.rep_learner(sents, pos, lengths, masks)
            event_flat_reps = event_reps.contiguous().view(-1, event_reps.size()[-1])
            event_flat_reps = event_flat_reps * masks.contiguous().view(-1, 1)
            event_outputs = self.event_extractor.classifier(event_flat_reps)
            pooled_event_rep = self.pooling_layer(event_reps.permute(0,2,1))
            event_domains = self.adv_classifier(gradreverse(pooled_event_rep.squeeze(-1), self.adv_coeff))

        return domain_outputs, event_outputs, event_domains


if __name__ == "__main__":
    x = Variable(torch.randn(5, 5).data, requires_grad=True)
    gradreverse = GradReverse.apply
    x = gradreverse(x, 2.0)
    print(x)
    z = torch.zeros(5, dtype=torch.int64)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(x, z)
    loss.backward()
    adversary = Adversary(100, 50, 2, 5)
    print(adversary)
    weights = nn.Linear(5, 100)
    x = weights(x)
    x = adversary.forward(x)
    loss = nn.CrossEntropyLoss()(x, torch.ones(5, dtype=torch.int64))
    loss.backward()
    optimizer = optim.Adam(adversary.parameters())
    optimizer.step()

    #Check that gradients don't backprop beyond classifier
