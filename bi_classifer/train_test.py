import torch
from tqdm.autonotebook import tqdm


from torch.nn import functional as F

from Cooperation_project.f1_score import f1score
from bi_classifer.f1_score import calc_accuracy


def train_test(EPOCHS=None, model=None, train_dataloader=None, test_dataloader=None, optimizer=None, log_interval=None,
          max_grad_norm=None, scheduler=None):
    for e_idx, epoch in enumerate(range(EPOCHS)):

        test_f1 = 0.0
        train_f1 = 0.0
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        for b_idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            X, y = batch
            loss = model.training_step(X, y)

            loss.backward()  # backprop the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()  # gradient step
            scheduler.step()  # resetting the gradients.

            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            # _, max_indices = torch.max(y_hat, 1)

            train_acc += calc_accuracy(y_hat, y)
            # train_f1 = f1score(y, max_indices)


            if (b_idx+1) % log_interval == 0:

                print("epoch {} batch id {} loss {} train acc {}".format(e_idx+1, b_idx + 1, loss.item(), train_acc / (b_idx+1)))

        print("epoch {} avg_loss {} train acc {} ".format(e_idx+1, loss.item(), train_acc / (b_idx+1)))

        model.eval()
        for b_idx, batch in enumerate(tqdm(test_dataloader)):
            X, y = batch
            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            # _, max_indices = torch.max(y_hat, 1)
            # test_f1 += f1score(y, max_indices)  # accuracy
            test_acc += calc_accuracy(y_hat, y)
        print(f"epoch {e_idx+1} test acc {test_acc/(b_idx+1)}" )

    torch.save(model, 'model.pth')