# import Pkg; Pkg.add("CUDA")
# import Pkg; Pkg.add("FileIO")
# import Pkg; Pkg.add("CairoMakie")
# import Pkg; Pkg.add("cuDNN")

using Flux, MLDatasets, CUDA, Random, Statistics
using Flux: onehotbatch, flatten, onecold, params
using CairoMakie
using FileIO

CUDA.allowscalar(false)
const USE_GPU = CUDA.functional()
DEV(x) = USE_GPU ? gpu(x) : x
Random.seed!(45)
BATCH = 128

onehot(y) = onehotbatch(y, 0:9)

function loaders()
    train_x, train_y = CIFAR10.traindata(Float32)
    test_x,  test_y  = CIFAR10.testdata(Float32)
    train_dl = Flux.DataLoader((train_x, onehot(train_y)); batchsize=BATCH, shuffle=true)
    test_dl  = Flux.DataLoader((test_x,  onehot(test_y));  batchsize=BATCH)
    return train_dl, test_dl
end

function accuracy(m, data)
    correct = 0
    total   = 0
    for (x, y) in data
        preds   = onecold(m(DEV(x)))
        labels  = onecold(y)
        correct += sum(preds .== labels)
        total   += length(labels)
    end
    return correct / total
end

function lenet5()
    Chain(
        Conv((5,5), 3=>6, relu),
        MaxPool((2,2)),
        Conv((5,5), 6=>16, relu),
        MaxPool((2,2)),
        flatten,
        Dense(16*5*5, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax,
    ) |> gpu
end

function lenet_k(k::Int)
    conv_part = Chain(
        Conv((k,k), 3=>6, relu),
        MaxPool((2,2)),
        Conv((k,k), 6=>16, relu),
        MaxPool((2,2)),
    )
    dummy  = conv_part(rand(Float32, 32,32,3,1))
    fc_in  = prod(size(dummy)[1:3])
    return Chain(
            conv_part,
            flatten,
            Dense(fc_in, 120, relu),
            Dense(120, 84, relu),
            Dense(84, 10),
            softmax
        ) |> gpu
end

function train!(model, train_dl, test_dl; epochs=5, lr=1e-3)
    opt = Flux.Adam(lr)
    for epoch in 1:epochs
        for (x,y) in train_dl
            x, y = gpu(x), gpu(y)
            gs = gradient(() -> Flux.crossentropy(model(x), y), params(model))
            Flux.Optimise.update!(opt, params(model), gs)
        end
        println("epoch $epoch | test acc = $(round(accuracy(model, test_dl)*100; digits=2)) %")
    end
end


# ---------------------------Question 1-------------------------------------------
train_dl, test_dl = loaders()
model_q1 = lenet5()
train!(model_q1, train_dl, test_dl; epochs=10)


# ---------------------------Question 2-------------------------------------------

unique_counts = [10_000, 20_000, 30_000]
epochs_per     = Dict(10_000=>6, 20_000=>3, 30_000=>2)
accs = Float64[]

full_x, full_y = CIFAR10.traindata(Float32)
for N in unique_counts
    e = epochs_per[N]
    println("\n>>> $N unique images | $e epoch(s)")
    idx   = randperm(50_000)[1:N]
    x_sub = full_x[:,:,:,idx]
    y_sub = onehot(full_y[idx])

    dl_small = Flux.DataLoader((x_sub, y_sub); batchsize=BATCH, shuffle=true)

    m = lenet5()
    train!(m, dl_small, test_dl; epochs=e)
    acc = accuracy(m, test_dl)
    push!(accs, acc)
    println("final test acc = $(round(acc*100; digits=2)) %")
end

fig = Figure(resolution = (450, 300))
ax  = Axis(
        fig[1, 1],
        xlabel = "#unique training images",
        ylabel = "test accuracy [%]",
        title  = "Diversity vs. repetition"
    )

ys = accs .* 100
xs = unique_counts

lines!(ax, xs, ys)
scatter!(ax, xs, ys; marker = :circle, markersize = 12)

save("q2_acc_vs_data.png", fig)

# The results show that accuracy fluctuates with the number of unique training images: around 46% for 10k, 
# dropping to 44% at 20k, then rising to 47% with 30k. Increasing from 10k to 20k introduces more diversity, 
# but each image is seen fewer times, leading to under-training and a dip in performance. At 30k, however, 
# there's a better trade-off—enough variety in the data while still repeating each sample twice—allowing the model to 
# learn more effectively. This highlights the importance of balancing data diversity and repetition when training with a 
# fixed number of gradient steps.

# -----------------------Question 3-----------------------------------------------
kernel_sizes = [3, 5, 7]
accs_k       = Float64[]

for k in kernel_sizes
    println("\n>>> kernel $k×$k")
    m = lenet_k(k)
    train!(m, train_dl, test_dl; epochs = 4)
    push!(accs_k, accuracy(m, test_dl))
    println("final test acc = $(round(accs_k[end]*100; digits=2)) %")
end

fig = Figure(resolution = (450, 300))
ax  = Axis(
        fig[1, 1],
        xlabel = "kernel size",
        ylabel = "test accuracy [%]",
        title  = "Effect of filter size on CIFAR-10"
        )

ys = accs_k .* 100
xs = kernel_sizes

lines!(ax, xs, ys)
scatter!(ax, xs, ys; marker = :circle, markersize = 12)

save("q3_kernel_vs_acc.png", fig)

# The results reveal that smaller convolutional kernels tend to perform better: 3x3 kernels achieved 
# the highest accuracy at around 56%, followed by 5x5 at 54%, while 7x7 lagged behind at 48%. 
# This is because smaller kernels enable deeper effective networks within the same parameter budget, 
# allowing the model to learn more complex and layered features. In contrast, larger kernels like 
# 7x7 consume capacity inefficiently by focusing on local patterns that earlier layers have likely 
# already captured, ultimately hindering generalization.

# --------------------------Question 4--------------------------------------------
img = test_dl.data[1][:,:,:,1]
img4d = reshape(img, size(img)..., 1)
# conv1 = model_q1.layers[1]
acts = cpu(model_q1.layers[1](DEV(img4d)))

fig = Figure(resolution=(300,300))
heatmap(fig[1,1], acts[:,:,1,1])
save("conv1_feature.png", fig)