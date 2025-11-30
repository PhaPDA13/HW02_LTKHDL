import numpy as np

class LogisticRegressionNumPy:
    def __init__(self, lr=0.05, momentum=0.9, max_epochs=5000, patience=80):
        self.lr = lr
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.patience = patience
    
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))
    
    def bce_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        self.w = np.zeros(X_train.shape[1])
        velocity = np.zeros_like(self.w)
        
        if X_val is not None:
            X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
        
        self.train_loss = []
        self.val_loss = []
        best_w = self.w.copy()
        best_val = np.inf
        wait = 0
        
        for epoch in range(self.max_epochs):
            z = np.einsum('ij,j->i', X_train, self.w)
            y_pred = self.sigmoid(z)
            loss = self.bce_loss(y_train, y_pred)
            self.train_loss.append(loss)
            
            if X_val is not None:
                z_val = np.einsum('ij,j->i', X_val, self.w)
                y_pred_val = self.sigmoid(z_val)
                val_l = self.bce_loss(y_val, y_pred_val)
                self.val_loss.append(val_l)
                
                if val_l < best_val - 1e-4:
                    best_val = val_l
                    best_w = self.w.copy()
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        print(f"   Early stopping tại epoch {epoch}")
                        self.w = best_w
                        break
            
            error = y_pred - y_train
            grad = np.einsum('i,ij->j', error, X_train) / len(y_train)
            velocity = self.momentum * velocity - self.lr * grad
            self.w += velocity
        
        if X_val is not None and wait < self.patience:
            self.w = best_w
        return self
    
    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        z = np.einsum('ij,j->i', X, self.w)
        proba = self.sigmoid(z)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int32)


def stratified_kfold_indices(y, n_splits=5, random_state=42):
    rng = np.random.default_rng(random_state)
    fraud = np.where(y == 1)[0]
    normal = np.where(y == 0)[0]
    rng.shuffle(fraud)
    rng.shuffle(normal)
    
    fraud_folds = np.array_split(fraud, n_splits)
    normal_folds = np.array_split(normal, n_splits)
    
    folds = []
    for i in range(n_splits):
        val_idx = np.concatenate([fraud_folds[i], normal_folds[i]])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        folds.append((train_idx, val_idx))
    return folds

