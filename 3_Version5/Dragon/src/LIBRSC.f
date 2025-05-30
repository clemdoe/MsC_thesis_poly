*DECK LIBRSC
      SUBROUTINE LIBRSC(MAXTRA,IPLIB,LBIN,NGRP,NBISO,ISONAM,MASKI,LSHI,
     1 NFS,IMPX,IALTER)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Compute the correlation information between a pair of resonant
* isotopes for the resonance spectrum expansion (RSE) method.
*
*Copyright:
* Copyright (C) 2023 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): A. Hebert
*
*Parameters: input
* MAXTRA  maximum number of energy bins of size DELI.
* IPLIB   pointer to the lattice microscopic cross section library
*         (L_LIBRARY signature).
* ISOT    index of the isotope been processed.
* LBIN    number of fine energy groups.
* NGRP    number of coarse energy groups.
* NBISO   number of isotopes present in the calculation domain.
* ISONAM  alias name of isotopes.
* MASKI   isotope masks (isotope with index I is process if
*         MASKI(I)=.true.).
* LSHI    resonant region number associated with each isotope.
* NFS     number of fine energy groups in each coarse energy group.
* IMPX    print flag (equal to zero for no print).
* IALTER  type of approximation (=0: use exponentials; =1: use Taylor
*         expansions).
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPLIB
      INTEGER MAXTRA,ISOT,LBIN,NGRP,NBISO,ISONAM(3,NBISO),LSHI(NBISO),
     1 NFS(NGRP),IMPX,IALTER
      LOGICAL MASKI(NBISO)
*----
*  LOCAL VARIABLES
*    KPLIB1: ISOTOPE WHERE THE COLLISION OCCURS
*    KPLIB2: SOURCE ISOTOPE
*----
      TYPE(C_PTR) KPLIB1,KPLIB2,LPLIB1,LPLIB2,MPLIB,IOFSET
      CHARACTER HNAMIS1*12,HNAMIS2*12,HSMG*131
*----
*  ALLOCATABLE ARRAYS
*----
      TYPE(C_PTR), ALLOCATABLE, DIMENSION(:) :: IPISO
      INTEGER, ALLOCATABLE, DIMENSION(:) :: NJJ,MRANK,NFS2,ISOMIX
      REAL, ALLOCATABLE, DIMENSION(:) :: EBIN,UUU,DEL,STR,SIGT,SIGS,PRI,
     1 STIS
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: TTT,DDD
      DOUBLE PRECISION, POINTER, DIMENSION(:,:) :: SSIGT,SSIGS
      LOGICAL, ALLOCATABLE, DIMENSION(:) :: LCORR
      TYPE MATRIX_ARRAY
        DOUBLE PRECISION, POINTER, DIMENSION(:,:) :: MATRIX
      END TYPE MATRIX_ARRAY
      TYPE(MATRIX_ARRAY), ALLOCATABLE, DIMENSION(:) :: SIGT_M,TSIGT_M,
     1 SCAT_M,DDD_M,U_M,T_M
      TYPE(MATRIX_ARRAY), ALLOCATABLE, DIMENSION(:,:) :: TSCAT_M
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(IPISO(NBISO),NJJ(NGRP),MRANK(NGRP),ISOMIX(NBISO),
     1 LCORR(NBISO))
      ALLOCATE(EBIN(LBIN+1),UUU(LBIN+1),DEL(LBIN),STR(LBIN),SIGT(LBIN),
     1 SIGS(LBIN),PRI(MAXTRA),STIS(LBIN))
      ALLOCATE(U_M(NGRP),T_M(NGRP),SIGT_M(NGRP),SCAT_M(NGRP),
     1 TSIGT_M(NGRP),TSCAT_M(NGRP,NGRP))
*----
*  FIND CORRELATED ISOTOPES.
*----
      CALL LIBIPS(IPLIB,NBISO,IPISO)
      CALL LCMGET(IPLIB,'ISOTOPESMIX',ISOMIX)
      LCORR(:NBISO)=.TRUE.
      DO 30 ISOT=1,NBISO
      IF(.NOT.MASKI(ISOT).OR.(LSHI(ISOT).EQ.0)) GO TO 30
      WRITE(HNAMIS1,'(3A4)') (ISONAM(I0,ISOT),I0=1,3)
      KPLIB1=IPISO(ISOT)
      DO 20 JSOT=1,NBISO
      IF(JSOT.EQ.ISOT) GO TO 20
      IF((.NOT.MASKI(JSOT)).OR.(LSHI(ISOT).NE.LSHI(JSOT))) GO TO 20
      WRITE(HNAMIS2,'(3A4)') (ISONAM(I0,JSOT),I0=1,3)
      IF(LSHI(ISOT).GT.0) THEN
        ! temperature correlation effect
        IF(HNAMIS2(:8).NE.HNAMIS1(:8)) GO TO 20
      ENDIF
      LCORR(ISOT)=.FALSE.
      CALL LCMSIX(KPLIB1,'PT-TABLE',1)
      CALL LCMLEN(KPLIB1,HNAMIS2,LENGT,ITYLCM)
      CALL LCMSIX(KPLIB1,' ',2)
      IF(LENGT.NE.0) GO TO 20
      IF(IMPX.GT.0) WRITE (6,'(/35H LIBRSC: COMPUTING CORRELATION EFFE,
     1 32HCTS BETWEEN ISOTOPES/MATERIALS '',A12,7H'' AND '',A12,2H''.)')
     2 HNAMIS1,HNAMIS2
*----
*  RECOVER ISOTOPIC AND AUTOLIB DATA.
*----
      ALLOCATE(NFS2(NGRP))
      KPLIB2=IPISO(JSOT)
      CALL LCMGET(KPLIB2,'AWR',AWR)
      CALL LCMGET(KPLIB2,'BIN-ENERGY',EBIN)
      CALL LCMGET(KPLIB2,'BIN-NTOT0',SIGT)
      CALL LCMLEN(KPLIB2,'BIN-SIGS00',ILONG,ITYLCM)
      CALL LCMGET(KPLIB2,'BIN-SIGS00',SIGS)
      CALL LCMGET(KPLIB2,'BIN-NFS',NFS2)
      IBIN=0
      DELMIN=1.0E10
      UUU(1)=0.0
      DO IG=1,NGRP
        IF(NFS(IG).NE.NFS2(IG)) THEN
          WRITE(HSMG,'(38HLIBRSC: INCOMPATIBLE BIN-NFS BETWEEN '',A12,
     1    7H'' AND '',A12,2H''.)') HNAMIS1,HNAMIS2
          CALL XABORT(HSMG)
        ENDIF
        DO LI=1,NFS(IG)
          DELM=LOG(EBIN(IBIN+LI)/EBIN(IBIN+LI+1))
          UUU(IBIN+LI+1)=LOG(EBIN(1)/EBIN(IBIN+LI+1))
          DEL(IBIN+LI)=DELM
          DELMIN=MIN(DELMIN,DELM)
        ENDDO
        IBIN=IBIN+NFS(IG)
      ENDDO
      DEALLOCATE(NFS2)
      IF(IBIN.NE.LBIN) CALL XABORT('LIBRSC: INVALID NUMBER OF BINS.')
      CALL LCMLEN(KPLIB2,'BIN-DELI',LENGT,ITYLCM)
      IF((LENGT.EQ.1).AND.(ITYLCM.EQ.2)) THEN
        CALL LCMGET(KPLIB2,'BIN-DELI',DELI)
      ELSE
        DELI=1.0/REAL(INT(1.00001/DELMIN))
      ENDIF
      PRI(:MAXTRA)=0.0
      CALL LIBPRI(MAXTRA,DELI,AWR,IALTER,0,NEXT,PRI)
*----
*  NULLIFY POINTERS
*----
      DO IG=1,NGRP
        NULLIFY(SIGT_M(IG)%MATRIX)
        NULLIFY(TSIGT_M(IG)%MATRIX)
        DO JG=1,NGRP
          NULLIFY(TSCAT_M(IG,JG)%MATRIX)
        ENDDO
      ENDDO
*----
* LOOP OVER RESONANT GROUPS AND RECOVER T_M AND U_M DATA
*----
      MRANK(:NGRP)=0
      CALL LCMSIX(KPLIB1,'PT-TABLE',1)
      CALL LCMGET(KPLIB1,'NOR',MRANK)
      LPLIB1=LCMGID(KPLIB1,'GROUP-RSE')
      DO IG=1,NGRP
        LGBIN=NFS(IG)
        IF(LGBIN.EQ.0) CYCLE
        MPLIB=LCMGIL(LPLIB1,IG)
        MI=MRANK(IG)
        CALL LCMGPD(MPLIB,'T_M',IOFSET)
        CALL C_F_POINTER(IOFSET,T_M(IG)%MATRIX,(/ MI,MI /))
        CALL LCMGPD(MPLIB,'U_M',IOFSET)
        CALL C_F_POINTER(IOFSET,U_M(IG)%MATRIX,(/ LGBIN,MI /))
      ENDDO
      CALL LCMSIX(KPLIB1,' ',2)
*----
* LOOP OVER RESONANT GROUPS
*----
      ALLOCATE(DDD_M(NGRP))
      LLL=0
      DO IG=1,NGRP
        NJJ(IG)=0
        LGBIN=NFS(IG)
        IF(LGBIN.EQ.0) CYCLE
        !
        ! compute SIGT_M
        MI=MRANK(IG)
        ALLOCATE(SSIGT(MI,MI))
        SSIGT(:,:)=0.0D0
        DO LI=1,LGBIN
          SIGMA=SIGT(LLL+LI)
          DO IMR1=1,MI
            DO IMR2=1,MI
              SSIGT(IMR1,IMR2)=SSIGT(IMR1,IMR2)+U_M(IG)%MATRIX(LI,IMR1)*
     1        SIGMA*U_M(IG)%MATRIX(LI,IMR2)
            ENDDO
          ENDDO
        ENDDO
        SIGT_M(IG)%MATRIX=>SSIGT
        NULLIFY(SSIGT)
        !
        ! compute SCAT_M
        DO JG=1,NGRP
          NULLIFY(SCAT_M(JG)%MATRIX)
          NULLIFY(DDD_M(JG)%MATRIX)
        ENDDO
        DO LI=1,LGBIN
          III=1
          CALL LIBECT(MAXTRA,LLL+LI,PRI,UUU(2),DELI,DEL,NEXT,III,MML,
     1    STIS)
          STR(:LBIN)=0.0
          DO MM=1,MML
            LLJ=LLL+LI-MM+1
            STR(LLJ)=STIS(MM)*SIGS(LLJ)
          ENDDO
          LLJ=LLL
          DO JG=IG,1,-1
            LGBIN2=NFS(JG)
            IF(LLL+LI-MML+1.GT.LLJ+LGBIN2) EXIT
            IF(.NOT.ASSOCIATED(U_M(JG)%MATRIX)) THEN
              CALL XABORT('LIBRSC: U_M(JG)%MATRIX IS NOT ASSOCIATED.')
            ENDIF
            MJ=MRANK(JG)
            IF(LI.EQ.1) THEN
              NJJ(IG)=NJJ(IG)+1
              ALLOCATE(DDD_M(JG)%MATRIX(LGBIN,MJ))
              DDD_M(JG)%MATRIX(:LGBIN,:MJ)=0.0D0
            ENDIF
            DO LJ=1,LGBIN2
              DDD_M(JG)%MATRIX(LI,:MJ)=DDD_M(JG)%MATRIX(LI,:MJ)+
     1        STR(LLJ+LJ)*U_M(JG)%MATRIX(LJ,:MJ)
            ENDDO
            IF(JG.GT.1) LLJ=LLJ-NFS(JG-1)
          ENDDO
        ENDDO
        !
        DO JG=IG-NJJ(IG)+1,IG
          IF(ASSOCIATED(U_M(IG)%MATRIX).AND.
     1       ASSOCIATED(DDD_M(JG)%MATRIX)) THEN
            MJ=MRANK(JG)
            ALLOCATE(SSIGS(MI,MJ))
            DO I=1,MI
              DO J=1,MJ
                SSIGS(I,J)=DOT_PRODUCT(U_M(IG)%MATRIX(:LGBIN,I),
     1          DDD_M(JG)%MATRIX(:LGBIN,J))
              ENDDO
            ENDDO
            SCAT_M(JG)%MATRIX => SSIGS
            NULLIFY(SSIGS)
            DEALLOCATE(DDD_M(JG)%MATRIX)
          ENDIF
        ENDDO
*----
*  LINEAR TRANSFORMATION
*----
        ALLOCATE(TTT(MI,MI))
        TTT(:MI,:MI)=TRANSPOSE(T_M(IG)%MATRIX(:MI,:MI))
        IF(ASSOCIATED(SIGT_M(IG)%MATRIX)) THEN
          ALLOCATE(TSIGT_M(IG)%MATRIX(MI,MI))
          ALLOCATE(SSIGT(MI,MI),DDD(MI,MI))
          DO I=1,MI
            DO J=1,MI
              DDD(I,J)=DOT_PRODUCT(SIGT_M(IG)%MATRIX(I,:MI),
     1        T_M(IG)%MATRIX(:MI,J))
            ENDDO
          ENDDO
          SSIGT=MATMUL(TTT,DDD)
          TSIGT_M(IG)%MATRIX => SSIGT
          NULLIFY(SSIGT)
          DEALLOCATE(DDD,SIGT_M(IG)%MATRIX)
        ENDIF
        DO JG=1,IG
          IF(ASSOCIATED(SCAT_M(JG)%MATRIX)) THEN
            MJ=MRANK(JG)
            ALLOCATE(TSCAT_M(IG,JG)%MATRIX(MI,MJ))
            ALLOCATE(SSIGS(MI,MJ),DDD(MI,MJ))
            DO I=1,MI
              DO J=1,MJ
                DDD(I,J)=DOT_PRODUCT(SCAT_M(JG)%MATRIX(I,:MJ),
     1          T_M(JG)%MATRIX(:MJ,J))
              ENDDO
            ENDDO
            SSIGS=MATMUL(TTT,DDD)
            TSCAT_M(IG,JG)%MATRIX => SSIGS
            NULLIFY(SSIGS)
            DEALLOCATE(DDD,SCAT_M(JG)%MATRIX)
          ENDIF
        ENDDO
        DEALLOCATE(TTT)
        LLL=LLL+LGBIN
      ENDDO
      DEALLOCATE(DDD_M)
*----
*  SAVE INFORMATION IN IPLIB
*----
      NPOS=0
      DO IG=1,NGRP
        NPOS=NPOS+NJJ(IG)
      ENDDO
      CALL LCMSIX(KPLIB1,'PT-TABLE',1)
      CALL LCMSIX(KPLIB1,HNAMIS2,1)
        LPLIB1=LCMLID(KPLIB1,'SIGT_M',NGRP) ! holds TSIGT_M information
        LPLIB2=LCMLID(KPLIB1,'SCAT_M',NPOS) ! holds TSCAT_M information
        CALL LCMPUT(KPLIB1,'NJJS00',NGRP,1,NJJ)
      CALL LCMSIX(KPLIB1,' ',2)
      IPOS=0
      DO IG=1,NGRP
        MI=MRANK(IG)
        IF(MI.EQ.0) CYCLE
        IF(ASSOCIATED(TSIGT_M(IG)%MATRIX)) THEN
          CALL LCMPDL(LPLIB1,IG,MI*MI,4,TSIGT_M(IG)%MATRIX)
          DEALLOCATE(TSIGT_M(IG)%MATRIX)
        ENDIF
        DO JG=IG,IG-NJJ(IG)+1,-1
          MJ=MRANK(JG)
          IPOS=IPOS+1
          IF(IPOS.GT.NPOS) CALL XABORT('LIBRSC: NPOS OVERFLOW.')
          IF(ASSOCIATED(TSCAT_M(IG,JG)%MATRIX)) THEN
            CALL LCMPDL(LPLIB2,IPOS,MI*MJ,4,TSCAT_M(IG,JG)%MATRIX)
            DEALLOCATE(TSCAT_M(IG,JG)%MATRIX)
          ELSE
            ALLOCATE(SSIGS(MI,MJ))
            SSIGS(:MI,:MJ)=0.0D0
            CALL LCMPDL(LPLIB2,IPOS,MI*MJ,4,SSIGS)
            DEALLOCATE(SSIGS)
          ENDIF
        ENDDO
      ENDDO
      CALL LCMSIX(KPLIB1,' ',2)
   20 CONTINUE
   30 CONTINUE
*----
*  ERASE T_M AND U_M DATA
*----
      DO 40 ISOT=1,NBISO
      IF(.NOT.MASKI(ISOT).OR.(LSHI(ISOT).EQ.0).OR.LCORR(ISOT)) GO TO 40
      KPLIB1=IPISO(ISOT)
      CALL LCMSIX(KPLIB1,'PT-TABLE',1)
      LPLIB1=LCMGID(KPLIB1,'GROUP-RSE')
      DO IG=1,NGRP
        IF(NFS(IG).EQ.0) CYCLE
        MPLIB=LCMGIL(LPLIB1,IG)
        CALL LCMDEL(MPLIB,'T_M')
        CALL LCMDEL(MPLIB,'U_M')
      ENDDO
      CALL LCMSIX(KPLIB1,' ',2)
   40 CONTINUE
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(TSCAT_M,TSIGT_M,SCAT_M,SIGT_M,T_M,U_M)
      DEALLOCATE(STIS,PRI,SIGS,SIGT,STR,DEL,UUU,EBIN)
      DEALLOCATE(LCORR,ISOMIX,MRANK,NJJ,IPISO)
      RETURN
      END
