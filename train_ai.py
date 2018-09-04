from torch.autograd import Variable

def train_ai(
        nstep_progress,
        moving_avg,
        eligibility_trace,
        save_brain,
        optimizer, loss,
        start_epoch, run_count,
        replay_memory, conv_network):

    for epoch in range(start_epoch, run_count + 1):
        replay_memory.run_steps(200)

        for batch in replay_memory.sample_batch(128):
            inputs, targets = eligibility_trace(conv_network, batch)
            inputs, targets = Variable(inputs), Variable(targets)
            predictions = conv_network(inputs)
            loss_error = loss(predictions, targets)

            optimizer.zero_grad()
            loss_error.backward()
            optimizer.step()


        rewards_steps = nstep_progress.rewards_steps()

        moving_avg.add(rewards_steps)

        avg_reward = moving_avg.average()

        save_brain(epoch, conv_network, optimizer, nstep_progress, moving_avg)

        print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
