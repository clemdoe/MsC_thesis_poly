*DECK MACDRV
      SUBROUTINE MACDRV(IPLIST,INDREC,IPRINT,IDF,NBMIX,NGROUP,NANISO,
     >                  NIFISS,NEDMAC,ITRANC,NDELG,NSTEP,NALBP)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Input macroscopic cross sections in Dragon/Donjon.
*
*Copyright:
* Copyright (C) 2006 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): G. Marleau
*
*Parameters: input
* IPLIST  LCM pointer to the macrolib.
* INDREC  =1 the macrolib is created;
*         =2 an existing macrolib is modified.
* IPRINT  print level.
* IDF     discontinuity factor flag.
*
*Parameters: output
* NBMIX   number of mixtures.
* NGROUP  maximum number of groups (default = 1).
* NANISO  maximum scattering anisotropy (default = 1 corresponding to
*         isotropic collision in laboratory).
* NIFISS  number of fissile isotopes per mixtures (default = 0).
* NEDMAC  number of aditional edition cross section types.
* ITRANC  type of transport correction: =0 no transport correction
*         =1 Apollo type transport correction; =2 recover from
*         library; =4 leakage correction alone.
* NDELG   number of precursor groups for delayed neutrons.
* NSTEP   number of delta cross-section sets used for generalized
*         perturbation theory (GPT) or kinetics calculations.
* NALBP   number of physical albedos.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPLIST
      INTEGER     INDREC,IPRINT,IDF,NBMIX,NGROUP,NANISO,NIFISS,NEDMAC,
     >            ITRANC,NDELG,NSTEP,NALBP
*----
*  LOCAL VARIABLES
*----
      TYPE(C_PTR) JPLIST,KPLIST,LPLIST
      PARAMETER  (IUNOUT=6,NCXST=18)
      CHARACTER   CARLIR*12,CARLU*4,CGOXSN*7
      LOGICAL     LIREAD,LOLDXS(NCXST),LNEWXS(NCXST),LNORM,LADD,LUPD
      INTEGER     ITYPLU,INTLIR,IMATER
      REAL        REALIR
      DOUBLE PRECISION DBLINP
      DOUBLE PRECISION SQFMAS,XDRCST,NMASS,EVJ
*----
* ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: ISCATA,IPERMU
      REAL, ALLOCATABLE, DIMENSION(:) :: TOTL,TOT1,FISS,SPEC,FIXE,TRANC,
     > DIFF,NFTOT,H,SCAT,NUDL,CHDL,OVERV,XSINT0,XSINT1,DIFFX,DIFFY,
     > DIFFZ,ENE,VEL,VOL
      REAL, ALLOCATABLE, DIMENSION(:,:) :: ALB
      CHARACTER(LEN=8), ALLOCATABLE, DIMENSION(:) :: HADF
      REAL, ALLOCATABLE, DIMENSION(:,:,:) :: XADF
*----
*  FOR AVERAGED NEUTRON VELOCITY
*  V=SQRT(2*ENER/M)=SQRT(2/M)*SQRT(ENER)
*  SQFMAS=SQRT(2/M) IN CM/S/SQRT(EV) FOR V IN CM/S AND E IN EV
*        =SQRT(2*1.602189E-19(J/EV)* 1.0E4(CM2/M2) /1.67495E-27 (KG))
*        =1383155.30602 CM/S/SQRT(EV)
*----
      EVJ=XDRCST('eV','J')
      NMASS=XDRCST('Neutron mass','kg')
      SQFMAS=SQRT(2.0D4*EVJ/NMASS)
*----
*  INITIALIZE USEFUL PARAMETERS
*----
      LIREAD=.TRUE.
      LNORM=.FALSE.
      LADD=.FALSE.
      LUPD=(INDREC.EQ.2)
      NEXTRE=1
      NEXTMI=1
      MAXFIS=MAX(1,NIFISS)
      ALLOCATE(ISCATA(NANISO))
      ISCATA(:NANISO)=0
      DO 120 IL=1,NCXST
        LOLDXS(IL)=.FALSE.
        LNEWXS(IL)=.FALSE.
 120  CONTINUE
      LPLIST=IPLIST
      ISTEP=0
      NTYPE=0
*----
*  READ A MAIN OPTION
*----
 1000 CALL REDGET(ITYPLU,INTLIR,REALIR,CARLIR,DBLINP)
      IF(ITYPLU.NE.3)
     >  CALL XABORT('MACDRV: READ ERROR 1 CHARACTER VARIABLE EXPECTED')
*----
*  CHECK FOR STOP/RETURN
*----
 1002 IF(CARLIR.EQ.';') THEN
        GO TO 2000
      ELSE IF(CARLIR.EQ.'EDIT') THEN
*----
*  READ THE PRINT INDEX
*----
        CALL REDGET(ITYPLU,IPRINT,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
      ELSE IF(CARLIR.EQ.'DELP') THEN
*----
*  READ THE NUMBER OF PRECURSOR GROUPS FOR DELAYED NEUTRONS
*----
        CALL REDGET(ITYPLU,NDELG,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
      ELSE IF(CARLIR.EQ.'STEP') THEN
*----
*  STEP TO A GPT SUB-DIRECTORY
*----
        CARLIR=' '
        CALL REDGET(ITYPLU,INTLIR,REALIR,CARLIR,DBLINP)
        IF((ITYPLU.NE.3).OR.(CARLIR.NE.'INPUT'))
     >  CALL XABORT('MACDRV: READ ERROR - INPUT STRING EXPECTED')
        CALL REDGET(ITYPLU,ISTEP,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
        IF(INDREC.EQ.1) THEN
          IF(ISTEP.LT.NSTEP)
     >    CALL XABORT('MACDRV: THIS DIRECTORY STEP ALREADY EXISTS.')
        ENDIF
        NSTEP=MAX(NSTEP,ISTEP)
        JPLIST=LCMLID(IPLIST,'STEP',NSTEP)
        KPLIST=LCMDIL(JPLIST,ISTEP)
        LPLIST=KPLIST
      ELSE IF(CARLIR.EQ.'READ') THEN
        IF(NBMIX.EQ.0) CALL XABORT('MACDRV: NBMIX NOT YET DEFINED')
        ALLOCATE(IPERMU(NBMIX))
        NBELEM=0
*----
*  IDENTIFY MIXTURES TO READ
*----
        NUMMAT=0
        DO 100 IMATER=1,NBMIX
          CALL REDGET(ITYPLU,JPERMU,REALIR,CGOXSN,DBLINP)
          IF(ITYPLU.NE.1) GO TO 1001
          IF(JPERMU.GT.NBMIX) CALL XABORT('MACDRV: MATERIAL NUMBER IS'
     >      //' LARGER THAN NBMIX')
          IPERMU(IMATER)=JPERMU
          NUMMAT=MAX(NUMMAT,JPERMU)
          IF(IMATER.EQ.NBMIX)
     >      CALL REDGET(ITYPLU,INTLIR,REALIR,CGOXSN,DBLINP)
          NBELEM=NBELEM+1
 100    CONTINUE
 1001   IF(ITYPLU.NE.3)
     >  CALL XABORT('MACDRV: READ ERROR 2 CHARACTER VARIABLE EXPECTED')
        IF(LIREAD) THEN
          ALLOCATE(TOTL(NBMIX*NGROUP),TOT1(NBMIX*NGROUP),
     >    FISS(NBMIX*NGROUP*MAXFIS),SPEC(NBMIX*NGROUP*MAXFIS),
     >    FIXE(NBMIX*NGROUP),TRANC(NBMIX*NGROUP),DIFF(NBMIX*NGROUP),
     >    NFTOT(NBMIX*NGROUP),H(NBMIX*NGROUP),
     >    SCAT(NBMIX*NGROUP*NGROUP*NANISO))
          ALLOCATE(NUDL(NBMIX*NGROUP*MAXFIS*MAX(NDELG,1)),
     >    CHDL(NBMIX*NGROUP*MAXFIS*MAX(NDELG,1)))
          ALLOCATE(OVERV(NBMIX*NGROUP),XSINT0(NBMIX*NGROUP),
     >    XSINT1(NBMIX*NGROUP),DIFFX(NBMIX*NGROUP),DIFFY(NBMIX*NGROUP),
     >    DIFFZ(NBMIX*NGROUP))
          CALL MACIXS(LPLIST,MAXFIS,NGROUP,NBMIX,NIFISS,NANISO,NDELG,
     >                TOTL,TOT1,FISS,SPEC,FIXE,TRANC,DIFF,NFTOT,H,SCAT,
     >                LOLDXS,ISCATA,NUDL,CHDL,DIFFX,DIFFY,DIFFZ,OVERV,
     >                XSINT0,XSINT1)
          LIREAD=.FALSE.
        ENDIF
        IF(CGOXSN.EQ.'INPUT  ') THEN
          ALLOCATE(HADF(NTYPE),XADF(NBMIX,NGROUP,NTYPE))
          CALL MACXSR(MAXFIS,NGROUP,NBMIX,NIFISS,NANISO,NDELG,NTYPE,
     >                TOTL,TOT1,FISS,SPEC,FIXE,TRANC,DIFF,NFTOT,H,SCAT,
     >                LOLDXS,LNEWXS,CARLIR,LADD,LUPD,IPRINT,ISCATA,
     >                NUDL,CHDL,DIFFX,DIFFY,DIFFZ,OVERV,XSINT0,XSINT1,
     >                HADF,XADF)
          IF(NTYPE.GT.0) THEN
            CALL LCMSIX(IPLIST,'ADF',1)
            CALL LCMPUT(IPLIST,'NTYPE',1,1,NTYPE)
            CALL LCMPTC(IPLIST,'HADF',8,NTYPE,HADF)
            DO ITYPE=1,NTYPE
              CALL LCMPUT(IPLIST,HADF(ITYPE),NBMIX*NGROUP,2,
     >        XADF(1,1,ITYPE))
            ENDDO
            CALL LCMSIX(IPLIST,' ',2)
          ENDIF
          DEALLOCATE(XADF,HADF)
        ENDIF
        DEALLOCATE(IPERMU)
        GO TO 1002
      ELSE IF(CARLIR.EQ.'NGRO') THEN
        CALL REDGET(ITYPLU,NGROUP,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
      ELSE IF(CARLIR.EQ.'NMIX') THEN
        CALL REDGET(ITYPLU,NBMIX,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
      ELSE IF(CARLIR.EQ.'ANIS') THEN
        NANISC=NANISO
        CALL REDGET(ITYPLU,NANISO,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
        IF(NANISO.GT.NANISC) THEN
          DEALLOCATE(ISCATA)
          ALLOCATE(ISCATA(NANISO))
          ISCATA(:NANISO)=0
        ENDIF
      ELSE IF(CARLIR.EQ.'NADF') THEN
         CALL REDGET(ITYPLU,NTYPE,REALIR,CARLIR,DBLINP)
         IF(ITYPLU.NE.1) CALL XABORT('MACDRV: READ ERROR - NUMBER ADF '
     >   //'TYPES EXPECTED.')
         IF(NTYPE.GT.0) IDF=2
      ELSE IF(CARLIR.EQ.'CTRA') THEN
*----
*  READ TRANSPORT CORRECTION TYPE
*----
        CALL REDGET(ITYPLU,INTLIR,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.3) CALL XABORT('MACDRV: READ ERROR - CHARACTER CT'
     >  //'RA TYPE EXPECTED')
        IF(CARLU.EQ.'NONE') THEN
          ITRANC=0
        ELSE IF(CARLU.EQ.'APOL') THEN
          ITRANC=1
        ELSE IF(CARLU.EQ.'WIMS') THEN
          ITRANC=2
        ELSE IF(CARLU.EQ.'LEAK') THEN
          ITRANC=4
        ELSE
          CALL XABORT('MACDRV: NONE, APOL, WIMS OR LEAK EXPECTED')
        ENDIF
      ELSE IF(CARLIR.EQ.'NIFI') THEN
        CALL REDGET(ITYPLU,NIFISS,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1)
     >  CALL XABORT('MACDRV: READ ERROR - INTEGER VARIABLE EXPECTED')
        MAXFIS=MAX(1,NIFISS)
      ELSE IF(CARLIR.EQ.'NORM') THEN
        LNORM=.TRUE.
      ELSE IF(CARLIR.EQ.'ADD') THEN
        IF(.NOT.LUPD) CALL XABORT('MACDRV: CANNOT USE THE ADD OPTION ON'
     >  //' A MACROLIB IN CREATION MODE')
        LADD=.TRUE.
      ELSE IF(CARLIR.EQ.'VOLUME') THEN
*----
*  READ MIXTURE VOLUMES
*----
          ALLOCATE(VOL(NBMIX))
          DO 177 IBM=1,NBMIX
            CALL REDGET(ITYPLU,INTLU,VOL(IBM),CARLU,DBLINP)
            IF(ITYPLU.NE.2) CALL XABORT('MACDRV: READ ERROR - REAL V'
     >      //'ARIABLE EXPECTED FOR VOLUME')
 177      CONTINUE
          CALL LCMPUT(IPLIST,'VOLUME',NBMIX,2,VOL)
          DEALLOCATE(VOL)
      ELSE IF(CARLIR.EQ.'ENER') THEN
*----
*  READ ENERGY GROUPS
*----
        IF(NGROUP.GT.0) THEN
          ALLOCATE(ENE(NGROUP+1),VEL(NGROUP))
          CALL REDGET(ITYPLU,INTLU,ENEMAX,CARLU,DBLINP)
          IF(ITYPLU.NE.2) CALL XABORT('MACDRV: READ ERROR - REAL VAR'
     >    //'IABLE EXPECTED')
          DO 179 IGR=1,NGROUP
            ENE(IGR)=ENEMAX
            CALL REDGET(ITYPLU,INTLU,ENECUR,CARLU,DBLINP)
            IF(ITYPLU.NE.2) CALL XABORT('MACDRV: READ ERROR - REAL V'
     >      //'ARIABLE EXPECTED')
            IF(ENECUR.GT.ENEMAX) CALL XABORT('MACDRV: READ ERROR - E'
     >      //'NERGY GOES FROM MAX TO MIN')
            ENEMAX=ENECUR
 179      CONTINUE
          IF(ENEMAX.LE.0.0) THEN
            ENE(NGROUP+1)=1.0E-5
          ELSE
            ENE(NGROUP+1)=ENEMAX
          ENDIF
          CALL LCMPUT(IPLIST,'ENERGY',NGROUP+1,2,ENE)
          VELG1=SQRT(ENE(1))
          JVEL=1
          DO 178 IGR=1,NGROUP
            VELG2=SQRT(ENE(IGR+1))
            VEL(JVEL)=REAL(SQFMAS)*SQRT(VELG1*VELG2)
            VELG1=VELG2
            ENE(IGR)=LOG(ENE(IGR)/ENE(IGR+1))
            JVEL=JVEL+1
 178      CONTINUE
          CALL LCMPUT(IPLIST,'AVGVEL',NGROUP,2,VEL)
          CALL LCMPUT(IPLIST,'DELTAU',NGROUP,2,ENE)
          DEALLOCATE(VEL,ENE)
        ENDIF
      ELSE IF(CARLIR.EQ.'ALBP') THEN
*----
*  READ GROUP INDEPENDENT PHYSICAL ALBEDOS
*----
        CALL REDGET(ITYPLU,NALBD,REALIR,CARLU,DBLINP)
        IF(ITYPLU.NE.1) CALL XABORT('MACDRV: INTEGER DATA EXPECTED.')
        IF(NALBD.GT.0) THEN
          NALBP=NALBD
          IF(NGROUP.EQ.0) CALL XABORT('MACDRV: NGROUP MISSING FOR ALBP')
          ALLOCATE(ALB(NALBD,NGROUP))
          DO IAL=1,NALBD
            DO IGR=1,NGROUP
              CALL REDGET(ITYPLU,INTLIR,ALB(IAL,IGR),CARLU,DBLINP)
              IF(ITYPLU.NE.2) CALL XABORT('MACDRV: ALBEDO EXPECTED.')
            ENDDO
          ENDDO
          CALL LCMPUT(IPLIST,'ALBEDO',NALBP*NGROUP,2,ALB)
          DEALLOCATE(ALB)
        ELSE
          NALBP=0
        ENDIF
      ELSE
        CALL XABORT('MACDRV: '//CARLIR//' IS AN INVALID KEY-WORD.')
      ENDIF
      GO TO 1000
*----
*  TRANSFER MODIFIED X-S ON THE MACROLIB
*----
 2000 IF(.NOT.LIREAD) THEN
        CALL MACPXS(LPLIST,MAXFIS,NGROUP,NBMIX,NIFISS,NANISO,NDELG,
     >              ITRANC,LNEWXS,TOTL,TOT1,FISS,SPEC,FIXE,TRANC,DIFF,
     >              NFTOT,H,SCAT,NEDMAC,ISCATA,NUDL,CHDL,DIFFX,DIFFY,
     >              DIFFZ,OVERV,XSINT0,XSINT1)
        DEALLOCATE(DIFFZ,DIFFY,DIFFX,XSINT1,XSINT0,OVERV)
        DEALLOCATE(CHDL,NUDL)
        DEALLOCATE(SCAT,H,NFTOT,DIFF,TRANC,FIXE,SPEC,FISS,TOT1,TOTL)
      ENDIF
      DEALLOCATE(ISCATA)
*----
*  NORMALIZATION OF X-S INFORMATION
*----
      IF(LNORM) THEN
        CALL MACNXS(IPLIST,MAXFIS,NGROUP,NBMIX,NIFISS,NANISO)
      ENDIF
*----
*  PRINT/CHECK X-S INFORMATION
*----
      IF((IPRINT.NE.0).AND.(IPRINT.NE.1)) THEN
        CALL MACWXS(IPLIST,IPRINT,NGROUP,NBMIX,NIFISS,NANISO,
     >              ITRANC,NEDMAC)
      ENDIF
      RETURN
      END
