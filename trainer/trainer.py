import torch
torch.set_default_tensor_type(torch.DoubleTensor)

def train_mlp(model, data_loader,loss_fn,optimizer,scheduler,**kargs):
    
    X_train,y_train=data_loader.get_train_data()
    
 

    loss_curve=[]
    # loop for max_iteration times
    for t in range(kargs["max_iteration"]):
        
        # renew optimizer
        optimizer.zero_grad(set_to_none=True)
        # forward propagate
        out, layer1_out, layer2_out= model(X_train)
  
        
        loss = loss_fn(out, y_train.reshape(-1,1))+kargs["lamda_1"]*(torch.cat([x.view(-1) for x in model.parameters()])**2).mean()
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