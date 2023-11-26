import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from model import Net


def cross_validation_with_val_set_p(dataset, args):
    assert args.epoch_select in ['val_max', 'test_max'], args.epoch_select

    model = Net(args.num_features, args.num_classes, args.hidden, args.num_gc_layers,
                 args.num_fc_layers, args.skip_connection, args.global_pool, args.dropout, args.alpha, args.flag, args.momentum).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    dataset = dataset.shuffle()
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optimizer, loader, args.device, args.flag)

        print('For suffix: {}, epoch: {}, pre-train loss: {:.4f}'.format(args.suffix, epoch, train_loss))

        if epoch % 20 == 0: 
            torch.save(model.state_dict(), '{}/{}_alpha={}_flag={}_lr={}_momentum={}_epoch={}_suffix={}_model.pt'.format(args.save_model, args.dataname, args.alpha, args.flag, args.lr, args.momentum, epoch, args.suffix))
            
    print("The pre-training of suffix {} have finished!".format(args.suffix))

    
def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device, flag):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        loss = model.forward_cl(data.x, data.edge_index, data.batch, device)
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        if flag == "GCN":
            model.update_ma()
    return total_loss / len(loader.dataset)






