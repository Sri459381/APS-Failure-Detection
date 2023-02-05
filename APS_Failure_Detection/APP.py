#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pickle
import streamlit as st

pickle_in = open("APS_FAILURE.pkl","rb")
classifier=pickle.load(pickle_in)

# Creating a function for prediction

def APS_PREDICTION (input_data):
    
    data = np.asarray(input_data)
    data_reshape = data.reshape(1,-1)
    prediction = loaded_model.predict(data_reshape)
    print(prediction)

    if (prediction[0] == 0):
        return "APS IS FAILURE"
    else:
        return "APS IS NOT FAILURE"

def main():
    #giving a title
    st.title("APS Failure Prediction")
    
    #getting the input data from the user
    aa_00 = st.text_input("Enter value for aa_00")
    ac_00 = st.text_input("Enter value for ac_00")
    ad_00 = st.text_input("Enter value for ad_00")
    ae_00 = st.text_input("Enter value for ae_00")
    af_00 = st.text_input("Enter value for af_00")
    ag_00 = st.text_input("Enter value for ag_00")
    ag_001 = st.text_input("Enter value for aa_001")
    ag_002 = st.text_input("Enter value for ag_002")
    ag_003 = st.text_input("Enter value for ag_003")
    ag_004 = st.text_input("Enter value for ag_004")
    ag_005 = st.text_input("Enter value for ag_005")
    ag_006 = st.text_input("Enter value for ag_006")
    ag_007 = st.text_input("Enter value for ag_007")
    ag_008 = st.text_input("Enter value for ag_008")
    ag_009 = st.text_input("Enter value for ag_000")
    ah_000 = st.text_input("Enter value for ah_000")
    ai_000 = st.text_input("Enter value for ai_000")
    aj_000 = st.text_input("Enter value for aj_000")
    ak_000 = st.text_input("Enter value for ak_000")
    al_000 = st.text_input("Enter value for al_000")
    am_0 = st.text_input("Enter value for am_0")
    an_000 = st.text_input("Enter value for an_000")
    am_000 = st.text_input("Enter value for am_000")
    ao_000 = st.text_input("Enter value for ao_000")
    ap_000 = st.text_input("Enter value for ap_000")
    aq_000 = st.text_input("Enter value for aq_000")
    ar_000 = st.text_input("Enter value for ar_000")
    as_000 = st.text_input("Enter value for as_000")
    at_000 = st.text_input("Enter value for at_000")
    au_000 = st.text_input("Enter value for au_000")
    av_000 = st.text_input("Enter value for av_000")
    ax_000 = st.text_input("Enter value for ax_000")
    ay_000 = st.text_input("Enter value for ay_000")
    ay_001 = st.text_input("Enter value for ay_001")
    ay_002 = st.text_input("Enter value for ay_002")
    ay_003 = st.text_input("Enter value for ay_003")
    ay_004 = st.text_input("Enter value for ay_004")
    ay_005 = st.text_input("Enter value for ay_005")
    ay_006 = st.text_input("Enter value for ay_006")
    ay_007 = st.text_input("Enter value for ay_007")
    ay_008 = st.text_input("Enter value for ay_008")
    ay_009 = st.text_input("Enter value for ay_009")
    az_000 = st.text_input("Enter value for az_000")
    az_001 = st.text_input("Enter value for az_001")
    az_002 = st.text_input("Enter value for az_002")
    az_003 = st.text_input("Enter value for az_003")
    az_004 = st.text_input("Enter value for az_004")
    az_005 = st.text_input("Enter value for az_005")
    az_006 = st.text_input("Enter value for az_006")
    az_007 = st.text_input("Enter value for az_007")
    az_008 = st.text_input("Enter value for az_008")
    az_009 = st.text_input("Enter value for az_009")
    ba_000 = st.text_input("Enter value for ba_000")
    ba_001= st.text_input("Enter value for ba_001")
    ba_002 = st.text_input("Enter value for ba_002")
    ba_003 = st.text_input("Enter value for ba_003")
    ba_004 = st.text_input("Enter value for ba_004")
    ba_005 = st.text_input("Enter value for ba_005")
    ba_006 = st.text_input("Enter value for ba_006")
    ba_007 = st.text_input("Enter value for ba_007")
    ba_008 = st.text_input("Enter value for ba_008")
    ba_009 = st.text_input("Enter value for ba_009")
    bb_000 = st.text_input("Enter value for bb_000")
    bc_000 = st.text_input("Enter value for bc_000")
    bd_000= st.text_input("Enter value for bd_000")
    be_000= st.text_input("Enter value for be_000")
    bf_000= st.text_input("Enter value for bf_000")
    bg_000= st.text_input("Enter value for bg_000")
    bh_000= st.text_input("Enter value for bh_000")
    bi_000= st.text_input("Enter value for bi_000")
    bj_000= st.text_input("Enter value for bj_000")
    bs_000= st.text_input("Enter value for bs_000")
    bt_000= st.text_input("Enter value for bt_000")
    bu_000= st.text_input("Enter value for bu_000")
    bv_000= st.text_input("Enter value for bv_000")
    bx_000= st.text_input("Enter value for bx_000")
    by_000= st.text_input("Enter value for by_000")
    bz_000= st.text_input("Enter value for bz_000")
    ca_000= st.text_input("Enter value for ca_000")
    cb_001 = st.text_input("Enter value for cb_000")
    cc_000= st.text_input("Enter value for cc_000")
    cd_000= st.text_input("Enter value for cd_000")
    ce_000 = st.text_input("Enter value for ce_000")
    cf_000 = st.text_input("Enter value for cf_000")
    cg_000 = st.text_input("Enter value for cg_000")
    ch_000 = st.text_input("Enter value for ch_000")
    ci_000 = st.text_input("Enter value for ci_000")
    cj_000 = st.text_input("Enter value for cj_000")
    ck_000 = st.text_input("Enter value for ck_000")
    cl_000 = st.text_input("Enter value for cl_000")
    cm_000 = st.text_input("Enter value for cm_000")
    cn_000 = st.text_input("Enter value for cn_000")
    cn_001 = st.text_input("Enter value for cn_001")
    cn_002 = st.text_input("Enter value for cn_002")
    cn_003 = st.text_input("Enter value for cn_003")
    cn_004 = st.text_input("Enter value for cn_004")
    cn_005 = st.text_input("Enter value for cn_005")
    cn_006 = st.text_input("Enter value for cn_006")
    cn_007 = st.text_input("Enter value for cn_007")
    cn_008 = st.text_input("Enter value for cn_008")
    cn_009 = st.text_input("Enter value for cn_009")
    co_000 = st.text_input("Enter value for co_000")
    cp_000 = st.text_input("Enter value for cp_000")
    cq_000 = st.text_input("Enter value for cq_000")
    cs_000 = st.text_input("Enter value for cs_000")
    cs_001 = st.text_input("Enter value for cs_001")
    cs_002 = st.text_input("Enter value for cs_002")
    cs_003 = st.text_input("Enter value for cs_003")
    cs_004 = st.text_input("Enter value for cs_004")
    cs_005 = st.text_input("Enter value for cs_005")
    cs_006 = st.text_input("Enter value for cs_006")
    cs_007 = st.text_input("Enter value for cs_007")
    cs_008 = st.text_input("Enter value for cs_008")
    cs_009 = st.text_input("Enter value for cs_009")
    ct_000 = st.text_input("Enter value for ct_000")
    cu_000 = st.text_input("Enter value for cu_000")
    cv_000 = st.text_input("Enter value for cv_000")
    cx_000 = st.text_input("Enter value for cx_000")
    cy_000 = st.text_input("Enter value for cy_000")
    cz_000 = st.text_input("Enter value for cz_000")
    da_000 = st.text_input("Enter value for da_000")
    db_000 = st.text_input("Enter value for db_000")
    dc_000 = st.text_input("Enter value for dc_000")
    dd_000 = st.text_input("Enter value for dd_000")
    de_000 = st.text_input("Enter value for de_000")
    df_000 = st.text_input("Enter value for df_000")
    dg_000 = st.text_input("Enter value for dg_000")
    dh_000 = st.text_input("Enter value for dh_000")
    di_000 = st.text_input("Enter value for di_000")
    dj_000 = st.text_input("Enter value for dj_000")
    dk_000 = st.text_input("Enter value for dk_000")
    dl_000 = st.text_input("Enter value for dl_000")
    dm_000 = st.text_input("Enter value for dm_000")
    dn_000 = st.text_input("Enter value for dn_000")
    do_000 = st.text_input("Enter value for do_000")
    dp_000 = st.text_input("Enter value for dp_000")
    dq_000 = st.text_input("Enter value for dq_000")
    dr_000 = st.text_input("Enter value for dr_000")
    ds_000 = st.text_input("Enter value for ds_000")
    dt_000 = st.text_input("Enter value for dt_000")
    du_000 = st.text_input("Enter value for du_000")
    dv_000 = st.text_input("Enter value for dv_000")
    dx_000 = st.text_input("Enter value for dx_000")
    dy_000 = st.text_input("Enter value for dy_000")
    dz_000 = st.text_input("Enter value for dz_000")
    ea_000 = st.text_input("Enter value for ea_000")
    eb_000 = st.text_input("Enter value for eb_000")
    ec_00 = st.text_input("Enter value for ec_00")
    ed_000 = st.text_input("Enter value for ed_000")
    ee_000 = st.text_input("Enter value for ee_000")
    ee_001 = st.text_input("Enter value for ee_001")
    ee_002= st.text_input("Enter value for ee_002")
    ee_003 = st.text_input("Enter value for ee_003")
    ee_004 = st.text_input("Enter value for ee_004")
    ee_005 = st.text_input("Enter value for ee_005")
    ee_006 = st.text_input("Enter value for ee_006")
    ee_007 = st.text_input("Enter value for ee_007")
    ee_008 = st.text_input("Enter value for ee_008")
    ee_009 = st.text_input("Enter value for ee_009")
    ef_000 = st.text_input("Enter value for ef_000")
    eg_000 = st.text_input("Enter value for eg_000")
   

    #Code for prediction
    result = ''
    
    # Creating a button for prediction
    if st.button("Predict"):
        result = APS_PREDICTION([aa_000,ac_000,ad_000,ae_000,af_000,ag_000,ag_001,ag_002,ag_003,ag_004,ag_005,
                                     ag_006,ag_007,ag_008,ag_009,ah_000,ai_000,aj_000,ak_000,al_000,am_0,an_000,ao_000,ap_000,
                                     aq_000,ar_000,as_000,at_000,au_000,av_000,ax_000,ay_000,ay_001,ay_002,ay_003,ay_004,ay_005,
                                     ay_006,ay_007,ay_008,ay_009,az_000,az_001,az_002,az_003,az_004,az_005,az_006,az_007,
                                     az_008,az_009,ba_000,ba_001,ba_002,ba_003,ba_004,ba_005,ba_006,ba_007,ba_008,ba_009,
                                     bb_000,bc_000,bd_000,be_000,bf_000,bg_000,bh_000,bi_000,bj_000,bs_000,bt_000,bu_000,
                                     bv_000,bx_000,by_000,bz_000,
                                     ca_000,cb_000,cc_000,cd_000,ce_000,cf_000,cg_000,ch_000,ci_000,cj_000,ck_000,cl_000,
                                     cm_000,cn_000,cn_001,cn_002,cn_003,cn_004,cn_005,cn_006,cn_007,cn_008,cn_009,co_000,
                                     cp_000,cq_000,cs_000,cs_001,cs_002,cs_003,cs_004,cs_005,cs_006,cs_007,cs_008,
                                     cs_009,ct_000,cu_000,cv_000,cx_000,cy_000,cz_000,da_000,db_000,dc_000,dd_000,de_000,
                                     df_000,dg_000,dh_000,di_000,dj_000,dk_000,dl_000,dm_000,dn_000,do_000,dp_000,dq_000,
                                     dr_000,ds_000,dt_000,du_000,dv_000,dx_000,dy_000,dz_000,ea_000,eb_000,ec_00,ed_000,
                                     ee_000,ee_001,ee_002,ee_003,ee_004,ee_005,ee_006,ee_007,ee_008,ee_009,ef_000,eg_000])
    
if __name__ == '__main__':
    main()
    


# In[ ]:




