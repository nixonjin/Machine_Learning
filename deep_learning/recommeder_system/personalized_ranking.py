# Bayesian Personalized Ranking Loss and its Implementation

from mexnet import gluon, np, npx
npx.set_np()

class BPRLoss(gluon.loss.Loss):
    def __init__(self, wight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weigth=None, batch_axis=0, **kwargs)
    
    def forward(self, positive, negative):
        distances = positive - negative
        loss = -np.sum(np.log(npx.sigmoid(distances)),0,keepdims=True)
        return loss

class HingeLossRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossRec, self).__init__(weigth=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(-distances + margin, 0))