# GradDescent

*Gradient Descent optimizers for Julia.*

## Introduction

This package abstracts the "boilerplate" code necessary for gradient descent. Gradient descent is "a way to minimize an objective function ``J(θ)`` parameterized by a model's parameters ``θ ∈ Rᵈ``" (Ruder 2017). Gradient descent finds ``θ`` which minizes ``J`` by iterating over the following update

``θ = θ - η ∇J(θ)``

until convergence of ``θ``. Certainly, the gradient calculation is model specific, however the learning rate ``η`` (at a given iteration) is not. Instead there are many different gradient descent variants which determine the learning rate. Each type of gradient descent optimizer has its own pros/cons. For most of these optimizers, the calculation of the learning rate is based on the value of the gradient (evaluated at a particular ``θ``) and a few (unrelated to the model) hyperparameters.

The purpose of this package is to allow the user to focus on the calculation of the gradients and not worry about the code for the gradient descent optimizer. I envision a user implementing his/her gradients, experimenting with various optimizers, and modifying the gradients as necessary.



## Optimizers

```@contents
Pages = ["optimizers.md"]
```
