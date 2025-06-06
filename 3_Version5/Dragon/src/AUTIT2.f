*DECK AUTIT2
      SUBROUTINE AUTIT2(IPTRK,IFTRAK,IPSYS,MAXTRA,KNORM,NUN,LBIN,NREG,
     1 NBMIX,NBISO,MAT,VOL,KEYFLX,NIRES,IAPT,CDOOR,LEAKSW,TITR,IMPX,
     2 CONC,SIGS,SIGT,SIGS1,DIL,PRI,UUU,DELI,ITRANC,NEXT,III,PHI)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Solution of the multigroup neutron flux for a non-pij method.
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
* IPTRK   pointer to the tracking (L_TRACK signature).
* IFTRAK  file unit number used to store the tracks.
* IPSYS   pointer to the system LCM object.
* MAXTRA  maximum number of elements in vector PRI.
* KNORM   type of cp normalization.
* NUN     number of unknowns in a single energy group.
* LBIN    number of energy groups.
* NREG    number of regions.
* NBMIX   number of mixtures in the internal library.
* NBISO   number of distinct isotopes.
* MAT     index-number of the mixture type assigned to each volume.
* VOL     volumes.
* KEYFLX  position of average fluxes in the unknown vector.
* NIRES   number of correlated resonant isotopes.
* IAPT    resonant isotope index associated with isotope I. Mixed
*         moderator if IAPT(I)=NIRES+1. Out-of-fuel isotope if
*         IAPT(I)=0.
* CDOOR   name of the geometry/solution operator.
* LEAKSW  leakage flag (LEAKSW=.true. if neutron leakage through
*         external boundary is present).
* TITR    title.
* IMPX    print flag (equal to zero for no print).
* CONC    number densities of each isotope in each mixture.
* SIGS    P0 scattering microscopic x-s.
* SIGT    total microscopic x-s.
* SIGS1   P1 scattering microscopic x-s.
* DIL     microscopic dilution cross section of each isotope.
* PRI     info to rebuild the SCAT matrix.
* UUU     lethargy limits of the groups.
* DELI    elementary lethargy width.
* ITRANC  type of transport correction.
* NEXT    used in subroutine LIBECT.
* III     offset in PRI array.
*
*Parameters: output
* PHI     neutron flux per unit lethargy.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPTRK,IPSYS
      INTEGER IFTRAK,MAXTRA,KNORM,NUN,LBIN,NREG,NBMIX,NBISO,MAT(NREG),
     1 NIRES,IAPT(NBISO),KEYFLX(NREG),IMPX,ITRANC,NEXT(NBISO),
     2 III(NBISO+1)
      REAL VOL(NREG),CONC(NBMIX,NBISO),SIGS(LBIN,NBISO),
     1 SIGT(LBIN,NBISO),SIGS1(LBIN,NBISO),DIL(NBISO),PRI(MAXTRA),
     2 UUU(LBIN+1),DELI,PHI(LBIN,NREG)
      LOGICAL LEAKSW
      CHARACTER CDOOR*12,TITR*72
*----
*  LOCAL VARIABLES
*----
      INTEGER, PARAMETER :: NGEFF=1
      TYPE(C_PTR) JPSYS,KPSYS,KPSOU1(NGEFF),KPSOU2(NGEFF)
      INTEGER NGIND(NGEFF),NBS2(NGEFF)
      CHARACTER HSMG*131
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: NLET,NPSYS
      REAL, ALLOCATABLE, DIMENSION(:) :: DEL,SOURCE,SIGTOT,SIGWIN,STIS,
     1 SUNKNO,FUNKNO
      REAL, ALLOCATABLE, DIMENSION(:,:) :: STR
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: Q
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(NLET(NBMIX),NPSYS(LBIN))
      ALLOCATE(DEL(LBIN),SOURCE(NREG),SIGTOT(0:NBMIX),SIGWIN(0:NBMIX),
     1 STR(LBIN,NBMIX),STIS(LBIN),SUNKNO(NUN),FUNKNO(NUN))
      ALLOCATE(Q(NREG))
*
      JPSYS=LCMLID(IPSYS,'GROUP',LBIN)
      DO 10 LLL=1,LBIN
      DEL(LLL)=UUU(LLL+1)-UUU(LLL)
   10 CONTINUE
      DO 60 LLL=1,LBIN
      NPSYS(LLL)=LLL
*----
*  COMPUTE THE TOTAL SCATTERING CROSS SECTIONS.
*----
      SIGTOT(0)=0.0
      DO 20 M=1,NBMIX
      SIGTOT(M)=0.0
      DO 15 K=1,NBISO
      IF(ITRANC.NE.0) SIGTOT(M)=SIGTOT(M)-CONC(M,K)*SIGS1(LLL,K)
      SIGTOT(M)=SIGTOT(M)+CONC(M,K)*(DIL(K)+SIGT(LLL,K))
   15 CONTINUE
   20 CONTINUE
      IF(IMPX.GE.9) THEN
         WRITE (6,'(//45H AUTIT2: TOTAL MACROSCOPIC CROSS SECTIONS IN ,
     1   5HGROUP,I5,1H:/)') LLL
         WRITE (6,'(1X,1P,10E13.5)') (SIGTOT(MAT(NRE)),NRE=1,NREG)
      ENDIF
*----
*  COMPUTE THE P0 WITHIN-GROUP SCATTERING CROSS SECTIONS.
*----
      SIGWIN(0:NBMIX)=0.0
      DO 50 K=1,NBISO
      IF((IAPT(K).GT.0).AND.(IAPT(K).LE.NIRES)) THEN
        CALL LIBECT(MAXTRA,LLL,PRI,UUU(2),DELI,DEL,NEXT(K),III(K),MML,
     1  STIS)
        DO 30 M=1,NBMIX
        SIGWIN(M)=SIGWIN(M)+CONC(M,K)*STIS(1)*SIGS(LLL,K)
   30   CONTINUE
      ENDIF
      IF(ITRANC.NE.0) THEN
        DO 40 M=1,NBMIX
        SIGWIN(M)=SIGWIN(M)-CONC(M,K)*SIGS1(LLL,K)
   40   CONTINUE
      ENDIF
   50 CONTINUE
      IF(IMPX.GE.10) THEN
         WRITE (6,'(//45H P0 WITHIN-GROUP SCATTERING MACROSCOPIC CROSS,
     1   18H SECTIONS IN GROUP,I5,1H:/)') LLL
         WRITE (6,'(1X,1P,10E13.5)') (SIGWIN(MAT(NRE)),NRE=1,NREG)
      ENDIF
*
      KPSYS=LCMDIL(JPSYS,LLL)
      CALL LCMPUT(KPSYS,'DRAGON-TXSC',NBMIX+1,2,SIGTOT(0))
      CALL LCMPUT(KPSYS,'DRAGON-S0XSC',NBMIX+1,2,SIGWIN(0))
   60 CONTINUE
*----
*  COMPUTE THE GROUPWISE RESPONSE MATRICES.
*----
      NANI=1
      NW=0
      NALBP=0
      ISTRM=1
      CALL DOORAV (CDOOR,JPSYS,NPSYS,IPTRK,IFTRAK,IMPX,LBIN,NREG,
     > NBMIX,NANI,NW,MAT,VOL,KNORM,LEAKSW,TITR,NALBP,ISTRM)
*----
*  COMPUTE THE ELASTIC SLOWING-DOWN SOURCES.
*----
      DO 160 LLL=1,LBIN
      DO M=1,NBMIX
        NLET(M)=1
        STR(:LBIN,M)=0.0
      ENDDO
      DO 90 K=1,NBISO
      IF((IAPT(K).GT.0).AND.(IAPT(K).LE.NIRES)) THEN
        CALL LIBECT(MAXTRA,LLL,PRI,UUU(2),DELI,DEL,NEXT(K),III(K),MML,
     1  STIS)
        DO 80 M=1,NBMIX
        AUX=CONC(M,K)
        IF(AUX.EQ.0.) GOTO 80
        NLET(M)=MAX(NLET(M),MML)
        DO 70 MM=1,MML
        LLJ=LLL-MM+1
        STR(MM,M)=STR(MM,M)+AUX*STIS(MM)*SIGS(LLJ,K)*DEL(LLJ)/DEL(LLL)
   70   CONTINUE
   80   CONTINUE
      ENDIF
   90 CONTINUE
*----
*  DILUTION SOURCE.
*----
      SOURCE(:NREG)=0.0
      DO 110 NRE=1,NREG
      IBM=MAT(NRE)
      IF(IBM.GT.0) THEN
        DO 100 K=1,NBISO
        IF((IAPT(K).EQ.0).OR.(IAPT(K).EQ.NIRES+1)) THEN
          SOURCE(NRE)=SOURCE(NRE)+CONC(IBM,K)*SIGS(LLL,K)
        ELSE
          SOURCE(NRE)=SOURCE(NRE)+CONC(IBM,K)*DIL(K)
        ENDIF
  100   CONTINUE
      ENDIF
  110 CONTINUE
*----
*  SCATTERING SOURCE.
*----
      DO 130 NRE=1,NREG
      Q(NRE)=SOURCE(NRE)
      M=MAT(NRE)
      IF(M.GT.0) THEN
         DO 120 MM=2,MIN(LLL,NLET(M))
         Q(NRE)=Q(NRE)+STR(MM,M)*PHI(LLL-MM+1,NRE)
  120    CONTINUE
      ENDIF
  130 CONTINUE
      IF(IMPX.GE.8) WRITE(6,'(7H GROUP=,I5,7H     S=,2X,1P,9D12.4/
     1 (21X,9D12.4))') LLL,(Q(NRE),NRE=1,NREG)
      SUNKNO(:NUN)=0.0
      FUNKNO(:NUN)=0.0
      DO 140 NRE=1,NREG
      SUNKNO(KEYFLX(NRE))=REAL(Q(NRE))
  140 CONTINUE
*----
*  FLUX SOLUTION.
*----
      IDIR=0
      NGIND(1)=LLL
      NBS2(1)=0
      KPSOU1(1)=C_NULL_PTR
      KPSOU2(1)=C_NULL_PTR
      IMPX2=MAX(0,IMPX-5)
      KPSYS=LCMDIL(JPSYS,LLL)
      IF(CDOOR.EQ.'SYBIL') THEN
         CALL SYBILF(KPSYS,IPTRK,IFTRAK,IMPX2,NGEFF,NGIND,IDIR,NREG,
     1               NUN,MAT,VOL,FUNKNO,SUNKNO,TITR)
      ELSE IF(CDOOR.EQ.'SN') THEN
         CALL SNF(KPSYS,IPTRK,IFTRAK,IMPX2,NGEFF,NGIND,IDIR,NREG,
     1            NBMIX,NUN,MAT,VOL,KEYFLX,FUNKNO,SUNKNO,TITR,NBS2,
     2            KPSOU1,KPSOU2)
      ELSE
         WRITE(HSMG,'(13HAUTIT2: DOOR ,A,20H IS NOT IMPLEMENTED.)')
     1   TRIM(CDOOR)
         CALL XABORT(HSMG)
      ENDIF
      DO 150 NRE=1,NREG
      PHI(LLL,NRE)=FUNKNO(KEYFLX(NRE))
  150 CONTINUE
      IF(IMPX.GE.8) WRITE(6,'(7H GROUP=,I5,7H  FLUX=,2X,1P,9E12.4/
     1 (21X,9E12.4))') LLL,(PHI(LLL,NRE),NRE=1,NREG)
  160 CONTINUE
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(Q,FUNKNO,SUNKNO,STIS,STR,SIGWIN,SIGTOT,SOURCE,DEL)
      DEALLOCATE(NPSYS,NLET)
      RETURN
      END
