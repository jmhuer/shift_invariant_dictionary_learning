import os

from sklearn.utils import shuffle
import torch
from torch.utils.tensorboard import SummaryWriter


def train_epoch(net, criterion, optimizer, dataset, routing_l1_regularization=0):
    total_loss = 0
    for i, batch in enumerate(dataset): 
        x_var = batch[:,0,0,:].to(net.device)
        x_var = x_var.to(net.device)
        net.to(net.device)

        optimizer.zero_grad()
        xpred_var = net(x_var)
        loss = criterion(xpred_var, x_var)
        if routing_l1_regularization:
            loss += routing_l1_regularization * torch.norm(net.routing_layer.weight, p=1)
        loss.backward(retain_graph=True)
        optimizer.step()
        loss = loss.item()
        total_loss += loss
        print("Epoch : {} \t Loss : {} ".format(i, round(loss,9)))
    return total_loss / i


# def log_losses(net, criterion, writer, X, Y, epoch, log_class_specific_losses=True):
#     running_losses = {'overall': 0}
#     running_counts = {'overall': 0}
#     if log_class_specific_losses:
#         for num in range(10):
#             running_losses[str(num)] = 0
#             running_counts[str(num)] = 0

#     with torch.no_grad():
#         for datum, label in zip(X, Y):
#             x_var = torch.FloatTensor(datum).unsqueeze(0)
#             x_var = x_var.to(net.device)
#             xpred_var = net(x_var)
#             loss = criterion(xpred_var, x_var).item()
#             running_losses['overall'] += loss
#             running_counts['overall'] += 1
#             if log_class_specific_losses:
#                 key = str(label.item())
#                 running_losses[key] += loss
#                 running_counts[key] += 1

#     for key, loss in running_losses.items():
#         writer.add_scalar(f'test_loss_{key}', loss / running_counts[key], epoch)
#     writer.flush()


# def log_activation_data(net, activation_writers, X_test, Y_test, num_stripes, epoch):
#     stripe_stats = net.get_stripe_stats(X_test, Y_test)
#     for stripe in range(num_stripes):
#         stripe_writer = activation_writers[stripe]
#         for digit in range(10):
#             stripe_writer.add_scalar(f'digit_{digit}', stripe_stats[digit][stripe], epoch)
#         stripe_writer.flush()


# def log_average_routing_scores(net, X, Y, writers, epoch):
#     running_scores = {}
#     running_counts = {}
#     for num in range(10):
#         running_scores[str(num)] = torch.zeros(net.num_stripes).to(net.device)
#         running_counts[str(num)] = 0

#     with torch.no_grad():
#         for datum, label in zip(X, Y):
#             x_var = torch.FloatTensor(datum).unsqueeze(0)
#             x_var = x_var.to(net.device)
#             digit = str(label.item())
#             running_scores[digit] += net.get_routing_scores(x_var).squeeze(0)
#             running_counts[digit] += 1

#     for stripe in range(net.num_stripes):
#         stripe_writer = writers[stripe]
#         for digit in range(10):
#             routing = running_scores[str(digit)][stripe].item() / running_counts[str(digit)]
#             stripe_writer.add_scalar(f'digit_routing_{digit}', routing, epoch)
#         stripe_writer.flush()


def train(net,
          criterion,
          optimizer,
          root_path,
          dataset,
          num_stripes,
          num_epochs,
          distort_prob_decay,
          routing_l1_regularization,
          log_class_specific_losses,
          should_log_average_routing_scores):
    main_writer = SummaryWriter(root_path)
    activation_writers = [SummaryWriter(os.path.join(root_path, str(num)))
                          for num in range(num_stripes)]

    for epoch in range(num_epochs):
        train_loss = train_epoch(net,
                                 criterion,
                                 optimizer,
                                 dataset["train"],
                                 routing_l1_regularization=routing_l1_regularization)
        net.distort_prob = max(net.distort_prob - distort_prob_decay, 0)
        main_writer.add_scalar('train_loss', train_loss, epoch)

        # log_losses(net,
        #            criterion,
        #            main_writer,
        #            X_test,
        #            Y_test,
        #            epoch,
        #            log_class_specific_losses=log_class_specific_losses)
        # log_activation_data(net,
        #                     activation_writers,
        #                     X_test,
        #                     Y_test,
        #                     num_stripes,
        #                     epoch)
        # if should_log_average_routing_scores:
        #     log_average_routing_scores(net,
        #                                X_test,
        #                                Y_test,
        #                                activation_writers,
        #                                epoch)

    main_writer.close()
    for writer in activation_writers:
        writer.close()
