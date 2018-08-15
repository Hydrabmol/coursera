function W = randInitializeWeights(epsilon_init, L_in, L_out)
  W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;