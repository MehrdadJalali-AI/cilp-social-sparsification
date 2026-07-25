# Dataset Audit

All methods must use these processed graphs, splits, seeds, and protocols.

## facebook

| Statistic | Value |
|---|---|
| citation | Rozemberczki et al., MUSAE / Facebook Page-Page (SNAP musae-facebook) |
| num_nodes | 22470 |
| directed_edge_entries | 341646 |
| unique_undirected_edges | 170823 |
| self_loops | 179 |
| duplicate_edges_approx | 0 |
| feature_dimension | 4714 |
| num_classes | 4 |
| class_distribution | `{0: 6495, 1: 6880, 2: 5768, 3: 3327}` |
| density | 0.0006766896339778271 |
| average_degree | 15.204539385847797 |
| num_connected_components | 1 |
| giant_component_ratio | 1.0 |
| homophily | 0.8853198885917664 |
| feature_sparsity | 0.9970300197601318 |
| assortativity | 0.08491250026368996 |
| clustering_coefficient | 0.3597383824426918 |
| library_versions | `{'torch': '2.4.0', 'torch_geometric': '2.6.1', 'networkx': '3.2.1', 'numpy': '1.26.4'}` |
| dataset_download_date | 2026-07-24 |
| raw_file_checksums | `{'edges.csv': '7c50d8f02a75cc0829577814a1fc14535164daa38d79c3612340c9e9cdbd4022', 'features.json': 'ea870537646a93642a0008d38aa9bfeff02018070ba97b1f3f469d9622626436', 'target.csv': '7bd96eafea3c2ca40f44bfa9e73642194696c21e38c7b29ed409c32ad14075cd'}` |
| preprocessing | `{'undirected': True, 'remove_self_loops': True, 'coalesce_duplicates': True, 'feature_normalization': 'none_at_audit', 'source': 'SNAP/MUSAE mirrors (PyG graphmining.ai unavailable)'}` |

## lastfm

| Statistic | Value |
|---|---|
| citation | Rozemberczki & Sarkar, LastFM Asia (SNAP) |
| num_nodes | 7624 |
| directed_edge_entries | 55612 |
| unique_undirected_edges | 27806 |
| self_loops | 0 |
| duplicate_edges_approx | 0 |
| feature_dimension | 7842 |
| num_classes | 18 |
| class_distribution | `{0: 1098, 1: 54, 2: 73, 3: 515, 4: 16, 5: 391, 6: 655, 7: 82, 8: 468, 9: 58, 10: 1303, 11: 138, 12: 57, 13: 63, 14: 570, 15: 257, 16: 254, 17: 1572}` |
| density | 0.0009568849118596328 |
| average_degree | 7.2943336831059815 |
| num_connected_components | 1 |
| giant_component_ratio | 1.0 |
| homophily | 0.87387615442276 |
| feature_sparsity | 0.949582040309906 |
| assortativity | 0.01707317256063152 |
| clustering_coefficient | 0.21941842432708525 |
| library_versions | `{'torch': '2.4.0', 'torch_geometric': '2.6.1', 'networkx': '3.2.1', 'numpy': '1.26.4'}` |
| dataset_download_date | 2026-07-24 |
| raw_file_checksums | `{'extracted/lasftm_asia/README.txt': '87376cef791bac0271775517e5a26678b22a525220832f0f00b6db207b92fb2e', 'extracted/lasftm_asia/lastfm_asia_edges.csv': '548eacff61bff564ab5007831fa4a7a30e1a3956075ffd24e43310ed33555842', 'extracted/lasftm_asia/lastfm_asia_features.json': '93fe84c96ffa9f0081ab37085979641062393895925de6c344df0f829c8ed2e7', 'extracted/lasftm_asia/lastfm_asia_target.csv': '91f19215f6eca8321feec038bd32900cf13fd4f7e041a9976cf202807c72ac16'}` |
| preprocessing | `{'undirected': True, 'remove_self_loops': True, 'coalesce_duplicates': True, 'feature_normalization': 'none_at_audit', 'source': 'SNAP/MUSAE mirrors (PyG graphmining.ai unavailable)'}` |

## github

| Statistic | Value |
|---|---|
| citation | Rozemberczki et al., GitHub Social (SNAP musae-github) |
| num_nodes | 37700 |
| directed_edge_entries | 578006 |
| unique_undirected_edges | 289003 |
| self_loops | 0 |
| duplicate_edges_approx | 0 |
| feature_dimension | 4005 |
| num_classes | 2 |
| class_distribution | `{0: 27961, 1: 9739}` |
| density | 0.0004066878203117068 |
| average_degree | 15.331724137931035 |
| num_connected_components | 1 |
| giant_component_ratio | 1.0 |
| homophily | 0.8453441858291626 |
| feature_sparsity | 0.9954277873039246 |
| assortativity | -0.07521713413904481 |
| clustering_coefficient | 0.16753704480107237 |
| library_versions | `{'torch': '2.4.0', 'torch_geometric': '2.6.1', 'networkx': '3.2.1', 'numpy': '1.26.4'}` |
| dataset_download_date | 2026-07-24 |
| raw_file_checksums | `{'extracted/git_web_ml/README.txt': '355e50a83b91eb7f34dc92dce991570f3e48c31df088eff7dd63dea32f764e6c', 'extracted/git_web_ml/citing.txt': '36016c1d9f1251594636dba9683198e9d0b0e70d11cbe47f991cc567dfbd0e70', 'extracted/git_web_ml/musae_git_edges.csv': '34c57382246949d1b3b7fa641a8532672001ecae8e9558f0b3c113cc035bd781', 'extracted/git_web_ml/musae_git_features.json': '47761f93f839d84b213e9f2fa43c056761734b295d6b2e838d8bbc335073cee7', 'extracted/git_web_ml/musae_git_target.csv': '3a8582f0c55d6f605df5f45bde28a90561be9517b40226c5e6bbce253679906a'}` |
| preprocessing | `{'undirected': True, 'remove_self_loops': True, 'coalesce_duplicates': True, 'feature_normalization': 'none_at_audit', 'source': 'SNAP/MUSAE mirrors (PyG graphmining.ai unavailable)'}` |
