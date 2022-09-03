import torch


def train(model, data_loader,loss_fn,optimizer,scheduler):
    
    X_train,y_train=data_loader.get_train_data()
    
    lamda_1=0.1
    max_iteration=3000
    
    loss_curve=[]
    # loop for max_iteration times
    for t in range(max_iteration):
        
        # renew optimizer
        optimizer.zero_grad(set_to_none=True)
        # forward propagate
        out, layer1_out, layer2_out= model(X_train)
  
        
        loss = loss_fn(out, y_train.reshape(-1,1))+lamda_1*(torch.cat([x.view(-1) for x in model.parameters()])**2).mean()
        if t % 100==0:
            print("loss:{},iter:{}".format(loss.item(),t))
        # record loss
        loss_curve.append(loss.item())
        
        
        
        loss.backward()
        
        # gradient descent
        optimizer.step()
        # learning rate decay
        scheduler.step()
    
    return model