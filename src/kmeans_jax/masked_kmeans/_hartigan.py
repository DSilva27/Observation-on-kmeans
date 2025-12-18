# from functools import partial
# from typing import Tuple

# import jax

#
# import jax.numpy as jnp
# from jaxtyping import Array, Bool, Float, Int

# from ._common_functions import assign_clusters, compute_loss, compute_centroids


# def compute_hartigan_distance(assignment, cluster_id, centroid, mask, averaged_mask):
#     perturbed_centroid = jax.lax.cond(jax.lax.cond)

#     return jax.lax.cond(
#         assignment == cluster_id,
#         lambda x: jax.lax.cond(
#             cluster_weight <= 1,
#             lambda x: 0.0,
#             lambda x: x * cluster_weight / (cluster_weight - 1),
#             distance,
#         ),
#         lambda x: x * cluster_weight / (cluster_weight + 1),
#         distance,
#     )


# def assign_dp_to_cluster_hartigan(centroids, labels, assignment_point, data_point):
#     cluster_populations = jnp.bincount(labels, length=centroids.shape[0])

#     distances = jnp.sum((data_point[None, ...] - centroids) ** 2, axis=-1)
#     distances = jax.vmap(weight_distance, in_axes=(None, 0, 0, 0))(
#         assignment_point, jnp.arange(centroids.shape[0]), cluster_populations, distances
#     )
#     return jnp.argmin(distances)


# def inner_loop_hartigan(labels, centroids, data):
#     def body_fun(i, val):
#         labels, centroids = val

#         assignment = assign_dp_to_cluster_hartigan(
#             centroids, labels, labels[i], data[i]
#         )

#         pred = labels[i] != assignment
#         labels = labels.at[i].set(assignment)

#         centroids = jax.lax.cond(
#             pred,
#             lambda x: compute_centroids(data, labels, centroids.shape[0]),
#             lambda x: x,
#             centroids,
#         )

#         return (labels, centroids)

#     return jax.lax.fori_loop(0, data.shape[0], body_fun, (labels, centroids))


# def _hartigan_kmeans_step(
#     carry: Tuple[
#         Float[Array, "K d"],
#         Int[Array, " n"],
#         Int[Array, " n"],
#         Float[Array, " max_steps"],
#         Int,
#     ],
#     data: Float[Array, "n d"],
# ) -> Tuple[
#     Float[Array, "K d"],
#     Int[Array, " n"],
#     Int[Array, " n"],
#     Float[Array, " max_steps"],
#     Int,
# ]:
#     centroids, old_labels, _, loss, counter = carry

#     labels, centroids = inner_loop_hartigan(
#         old_labels.copy(), centroids, data
#     )
#     # loss = loss.at[counter].set(compute_loss(data, centroids, labels))
#     loss = compute_loss(data, centroids, labels)
#     return (centroids, labels, old_labels, loss, counter + 1)


# def _hart_kmeans_stop_condition(
#     carry: Tuple[
#         Float[Array, "K d"],
#         Int[Array, " n"],
#         Int[Array, " n"],
#         Float[Array, " max_steps"],
#         Int,
#     ],
#     max_steps: Int,
# ) -> Bool:
#     _, labels, old_labels, _, counter = carry

#     cond1 = jnp.any(labels != old_labels)
#     cond2 = counter <= max_steps

#     return cond1 & cond2


# def run_hartigan_kmeans(
#     data: Float[Array, "n d"],
#     init_centroids: Float[Array, "K d"],
#     max_iters: Int = 1000,
# ) -> Tuple[
#     Tuple[Float[Array, "K d"], Int[Array, " n"]],
#     Float[Array, " max_steps"],
# ]:
#     loss = 0.0  # jnp.zeros(max_iters)
#     counter = 0

#     init_labels = assign_clusters(init_centroids, data)
#     init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])

#     cond_fun = jax.jit(partial(_hart_kmeans_stop_condition, max_steps=max_iters))

#     # makes sure the initial assignment does not trigger the stop condition
#     carry = (init_centroids, init_labels, init_labels - 1, loss, counter)

#     @jax.jit
#     def run_batched_hartigan_inner(carry):
#         return jax.lax.while_loop(
#             cond_fun=cond_fun,
#             body_fun=lambda c: _hartigan_kmeans_step(c, data),
#             init_val=carry,
#         )

#     centroids, labels, _, loss, counter = run_batched_hartigan_inner(carry)
#     return centroids, labels, loss, counter
