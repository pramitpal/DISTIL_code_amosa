from supporting_functions import *
###===========================================
if __name__ == "__main__":
    # Define networks
    model_dict = define_networks()
    # Core simulation parameters
    NPE = 40
    NT = 16
    xbar_size = 128
    SRAM_KB_per_tile = 72
    XBAR_bits_per_cell = 2
    Vmem_res = 8
    Timestep = 5
    # Model selection
    # 'vgg19' 4x4
    # 'resnet50' 4x4
    # 'densenet121' 5x5
    # 'LVVITB' 4x4
    # 'VIT'6x6
    # 'SPIKFORMER'5x5
    MODEL_NAME = 'resnet50'
    network = model_dict[MODEL_NAME]
    # NoC/NoI parameters
    NoC_buswidth = 16
    NoI_buswidth = 128
    NoC_wire_length = 0.14
    NoI_wire_length = 1.15
    NoC_Freq = 4e9
    NoI_Freq = 1e9
    NoC_cycle_time = 1e3 / NoC_Freq
    NoI_cycle_time = 1e3 / NoI_Freq
    # AMOSA parameters
    max_num_groups = 25
    iterations_per_temp = 50
    inter_cost=(NoC_Freq*NoC_buswidth*NoI_wire_length)/ (NoI_Freq*NoI_buswidth*NoC_wire_length)
    w_mem = 0.5
    N = 1
    # Performance and energy parameters
    TOPS = 27
    ENERGY_PER_MAC_pj = 0.3
    tile_area = 0.25
    chiplet_area = 4
    # System configuration
    mesh_rows, mesh_cols = 4, 4
    LIF_T_list = [6]
    # LIF_T_list = range(1, NT)
    # Traffic and DRAM parameters
    percent_keep = 0.99
    min_traffic = 2
    eta = 0.7
    DRAM_BW = 5

    # ============================================
    global_regular_df, global_optimized_df,final_summary_df = pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
    all_pareto_results, balanced_results_optimized,balanced_results_optimized,distil_results_by_lif = {}, {},{},{}
    # ============================================
    # Run global LIF case
    global_regular_df, global_optimized_df = run_global_lif_case(
        network=network, NPE=NPE, NT=NT, xbar_size=xbar_size, SRAM_KB_per_tile=SRAM_KB_per_tile, \
        XBAR_bits_per_cell=XBAR_bits_per_cell, Vmem_res=Vmem_res, Timestep=Timestep, \
        NoC_buswidth=NoC_buswidth, NoI_buswidth=NoI_buswidth, NoC_wire_length=NoC_wire_length, \
        NoI_wire_length=NoI_wire_length, NoC_Freq=NoC_Freq, NoI_Freq=NoI_Freq, TOPS=TOPS, \
        ENERGY_PER_MAC_pj=ENERGY_PER_MAC_pj, tile_area=tile_area, chiplet_area=chiplet_area, \
        mesh_rows=mesh_rows, mesh_cols=mesh_cols, LIF_T_list=LIF_T_list, percent_keep=percent_keep, \
        min_traffic=min_traffic, eta=eta, DRAM_BW=DRAM_BW,run_optimized=True)
    cleanup_booksim_files()
    
    # Run AMOSA optimization
    all_pareto_results, balanced_results = run_amosa_optimization(
        network=network,LIF_T_list=LIF_T_list,NPE=NPE,NT=NT,xbar_size=xbar_size,SRAM_KB_per_tile=SRAM_KB_per_tile,\
        XBAR_bits_per_cell=XBAR_bits_per_cell,Vmem_res=Vmem_res,Timestep=Timestep,NoC_buswidth=NoC_buswidth,\
        NoI_buswidth=NoI_buswidth,mesh_rows=mesh_rows,mesh_cols=mesh_cols,max_num_groups=max_num_groups,\
        iterations_per_temp=iterations_per_temp,inter_cost=inter_cost,w_mem=w_mem,N=N,\
        debug=0,debug_here=True)
    cleanup_booksim_files()
    # Optimize LIF placements for balanced solutions
    balanced_results_optimized = optimize_lif_placements(network=network,balanced_results=balanced_results,NT=NT,\
        NPE=NPE,xbar_size=xbar_size,XBAR_bits_per_cell=XBAR_bits_per_cell,\
        Vmem_res=Vmem_res,Timestep=Timestep,mesh_cols=mesh_cols,mesh_rows=mesh_rows,visualize=False,\
        NoC_buswidth=NoC_buswidth,NoI_buswidth=NoI_buswidth,debug=0)
    # Run with optimized LIF placement results
    distil_results_by_lif, final_summary_df = run_distil_optimized_lif(
        balanced_results_optimized=balanced_results_optimized,
        network=network, xbar_size=xbar_size, Vmem_res=Vmem_res,
        XBAR_bits_per_cell=XBAR_bits_per_cell, NT=NT, NPE=NPE,
        LIF_T_list=LIF_T_list, mesh_rows=mesh_rows, mesh_cols=mesh_cols,
        SRAM_KB_per_tile=SRAM_KB_per_tile, Timestep=Timestep,
        NoI_buswidth=NoI_buswidth, NoC_buswidth=NoC_buswidth,
        NoC_cycle_time=NoC_cycle_time, NoI_cycle_time=NoI_cycle_time,
        TOPS=TOPS, ENERGY_PER_MAC_pj=ENERGY_PER_MAC_pj,
        chiplet_area=chiplet_area, tile_area=tile_area,
        percent_keep=percent_keep, min_traffic=min_traffic,
        eta=eta, DRAM_BW=DRAM_BW, max_lif_tiles=1e3,
        visualize=False, inter_cost=inter_cost, debug=1)
    cleanup_booksim_files()
    
    
    # ============================================
    # SHOW AMOSA RESULT SUMMARY FOR BALANCED SOLUTIONS
    # ============================================
    # print("\n" + "-" * 70+"\n" + " " * 7+f" AMOSA OPTIMIZED SOLUTIONS - PERFORMANCE METRICS({MODEL_NAME})"+ "\n" + "-" * 70)
    # amosa_cols = ['LIF_T', 'num_groups', 'total_lif_tiles', 'total_cost']
    # all_dfs = [df[amosa_cols] for df in balanced_results_optimized.values() if not df.empty]
    # if all_dfs:
    #     print(pd.concat(all_dfs, ignore_index=True).to_string(index=False))
    # else:
    #     print("No results available")
    # ============================================
    # SHOW GLOBAL RESULT SUMMARY FOR REGULAR AND OPTIMIZED
    # ============================================
    print("\n" + "-" * 70+"\n" + " " * 15+f" GLOBAL - PERFORMANCE METRICS({MODEL_NAME}) "+ "\n" + "-" * 70)
    cols = ['lif_t', 'End_to_end_latency_ms', 'TOPS/Area', 'Total_area_sq_mm', 'LIF_MEM(KB)']
    print(global_regular_df[cols].to_string(index=False) if not global_regular_df.empty else "No results available")
    print("\n" + "-" * 70+"\n" + " " * 15+f" GLOBAL OPTIMIZED - PERFORMANCE METRICS({MODEL_NAME}) "+" " * 15+"\n" + "-" * 70)
    print(global_optimized_df[cols].to_string(index=False) if global_optimized_df is not None and not global_optimized_df.empty else "No results available")
    # ============================================
    # SHOW DISTIL RESULT SUMMARY FOR BALANCED SOLUTIONS
    # ============================================
    print("\n" + "-" * 70+"\n" + " " * 7+f" DISTIL RESULTS - PERFORMANCE METRICS({MODEL_NAME})"+ "\n" + "-" * 70)
    amosa_cols = ['lif_t', 'End_to_end_latency_ms','TOPS/Area','num_groups','Total_area_sq_mm','LIF_MEM(KB)']
    all_dfs = [df[amosa_cols] for df in distil_results_by_lif.values() if not df.empty]
    if all_dfs:
        print(pd.concat(all_dfs, ignore_index=True).to_string(index=False))
    else:
        print("No results available")
