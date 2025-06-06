*DECK EDIHFD
      SUBROUTINE EDIHFD(IPRINT,NGROUP,NGCOND,NREGIO,NMERGE,NBISO,NDEPL,
     >                  NDFI,NDFP,NREAC,MATCOD,VOLUME,INADPL,ISONAM,
     >                  ISONRF,IPISO,MIX,FLUXES,DEN,IDEPL,IGCOND,IMERGE,
     >                  KDRI,RRD,FIYI,DECAY,YIELD,FIPI,FIFP,PYIELD)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Recover depletion information from the reference internal library.
*
*Copyright:
* Copyright (C) 2002 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): A. Hebert
*
*Parameters: input
* IPRINT  print index.
* NGROUP  number of groups.
* NGCOND  number of condensed groups.
* NREGIO  number of regions.
* NMERGE  number of merged regions.
* NBISO   number of isotopes in the microlib.
* NDEPL   number of depleting isotopes.
* NDFI    number of direct fissile isotopes.
* NDFP    number of direct fission products.
* NREAC   number of depletion reactions.
* MATCOD  material per region.
* VOLUME  volume of region.
* INADPL  name of depleting isotopes.
* ISONAM  isotopes names.
* ISONRF  library name of isotopes.
* IPISO   pointer array towards microlib isotopes.
* MIX     mixture associated with isotopes.
* FLUXES  multigroup fluxes.
* DEN     isotope density.
* IDEPL   non depleting flag (=1 to stop depletion).
* IGCOND  limits of condensed groups.
* IMERGE  index of merged region.
* KDRI    depletion identifiers.
* RRD     radioactive decay constants.
* FIYI    fission yields.
*
*Parameters: output
* DECAY   radioactive decay constants for saves isotopes.
* YIELD   condensed fission product yield  (group ordered).
* FIPI    fissile isotope index assigned to each microlib isotope.
* FIFP    fission product index assigned to each microlib isotope.
* PYIELD  condensed fission product yield (fissile isotope ordered).
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPISO(NBISO)
      INTEGER     IPRINT,NGROUP,NGCOND,NREGIO,NMERGE,NBISO,NDEPL,
     >            NDFI,NDFP,NREAC,MATCOD(NREGIO),INADPL(3,NDEPL),
     >            ISONAM(3,NBISO),ISONRF(3,NBISO),MIX(NBISO),
     >            IDEPL(NBISO),IGCOND(NGCOND),IMERGE(NREGIO),
     >            KDRI(NREAC,NDEPL),FIPI(NBISO,NMERGE),
     >            FIFP(NBISO,NMERGE)
      REAL        VOLUME(NREGIO),FLUXES(NREGIO,NGROUP),DEN(NBISO),
     >            RRD(NDEPL),FIYI(NDFI,NDFP),DECAY(NBISO),
     >            YIELD(NGCOND+1,NDFP,NMERGE),PYIELD(NDFI,NDFP,NMERGE)
*----
*  LOCAL VARIABLES
*----
      TYPE(C_PTR) KPLIB
      INTEGER     IGAR(3)
      CHARACTER   HNISOR*12,TEXT12*12,HSMG*131
      LOGICAL     L1,L2
      DOUBLE PRECISION GAR
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: INDX
      REAL, ALLOCATABLE, DIMENSION(:) :: SIG
      REAL, ALLOCATABLE, DIMENSION(:,:) :: FIRA
      CHARACTER(LEN=12), ALLOCATABLE, DIMENSION(:) :: HDFI,HDFP
*----
*  SCRATCH STORAGE ALLOCATION
*   SIG     fission cross sections.
*   FIRA    fission rates.
*   INDX    depleting isotope index.
*----
      ALLOCATE(INDX(NBISO))
      ALLOCATE(SIG(NGROUP),FIRA(NGCOND+1,NMERGE))
*----
*  COMPUTE THE DEPLETING ISOTOPE INDEX
*----
      DO 20 ISO=1,NBISO
        IF(IDEPL(ISO).NE.1) THEN
          WRITE(HNISOR,'(3A4)') (ISONRF(I0,ISO),I0=1,3)
          I1=INDEX(HNISOR,'_')
          IF(I1.EQ.0) THEN
            TEXT12=HNISOR
          ELSE
            TEXT12=HNISOR(:I1-1)
          ENDIF
          READ(TEXT12,'(3A4)') (IGAR(I0),I0=1,3)
          DO 10 IDP=1,NDEPL
            L1=((ISONRF(1,ISO).EQ.INADPL(1,IDP)).AND.
     1          (ISONRF(2,ISO).EQ.INADPL(2,IDP)).AND.
     2          (ISONRF(3,ISO).EQ.INADPL(3,IDP)))
            L2=((IGAR(1).EQ.INADPL(1,IDP)).AND.
     1          (IGAR(2).EQ.INADPL(2,IDP)).AND.
     2          (IGAR(3).EQ.INADPL(3,IDP)))
            IF(L1.OR.L2) THEN
              INDX(ISO)=IDP
              GO TO 20
            ENDIF
  10      CONTINUE
        ENDIF
        INDX(ISO)=0
  20  CONTINUE
*----
*  MAIN ISOTOPIC LOOP
*----
      FIPI(:NBISO,:NMERGE)=0
      FIFP(:NBISO,:NMERGE)=0
      PYIELD(:NDFI,:NDFP,:NMERGE)=0.0
      YIELD(:NGCOND+1,:NDFP,:NMERGE)=0.0
      FIRA(:NGCOND+1,:NMERGE)=0.0
      DO 100 ISO=1,NBISO
        IDPL=INDX(ISO)
        IF(IDPL.EQ.0) GO TO 100
        KPLIB=IPISO(ISO) ! set ISO-th isotope
        IF(.NOT.C_ASSOCIATED(KPLIB)) THEN
          WRITE(HSMG,'(17HEDIHFD: ISOTOPE '',3A4,16H'' IS NOT AVAILAB,
     >    19HLE IN THE MICROLIB.)') (ISONAM(I0,ISO),I0=1,3)
          CALL XABORT(HSMG)
        ENDIF
*----
*  SET RADIOACTIVE DECAY CONSTANT
*----
        DECAY(ISO)=RRD(IDPL)
*----
*  COMPUTE CONDENSED FISSION RATES.
*----
        IF(MOD(KDRI(2,IDPL),100).EQ.4) THEN
          IFI=KDRI(2,IDPL)/100
          CALL LCMGET(KPLIB,'NUSIGF',SIG)
          DO 90 IREG=1,NREGIO
            IMR=IMERGE(IREG)
            IF((IMR.GT.0).AND.(MATCOD(IREG).EQ.MIX(ISO))) THEN
              FIPI(ISO,IMR)=IFI
              IGRFIN=0
              DO 80 IGC=1,NGCOND
                IGRDEB=IGRFIN+1
                IGRFIN=IGCOND(IGC)
                GAR=0.0D0
                DO 60 IGR=IGRDEB,IGRFIN
                  GAR=GAR+FLUXES(IREG,IGR)*DEN(ISO)*VOLUME(IREG)*
     >            SIG(IGR)
  60            CONTINUE
                DO 70 JSO=1,NBISO
                  JDPL=INDX(JSO)
                  IF(JDPL.EQ.0) GO TO 70
                  IF(MOD(KDRI(2,JDPL),100).EQ.5) THEN
                    ISOFP=KDRI(2,JDPL)/100
                    IF(ISOFP.EQ.0) CALL XABORT('EDIHFD: ISOFP.EQ.0.')
                    IF(ISOFP.GT.NDFP) CALL XABORT('EDIHFD: YIELD OVERF'
     >              //'LOW.')
                    FIFP(JSO,IMR)=ISOFP
                    DELTA=REAL(GAR)*FIYI(IFI,ISOFP)
                    YIELD(1,ISOFP,IMR)=YIELD(1,ISOFP,IMR)+DELTA
                    YIELD(IGC+1,ISOFP,IMR)=YIELD(IGC+1,ISOFP,IMR)+DELTA
                    PYIELD(IFI,ISOFP,IMR)=FIYI(IFI,ISOFP)
                  ENDIF
  70            CONTINUE
                FIRA(1,IMR)=FIRA(1,IMR)+REAL(GAR)
                FIRA(IGC+1,IMR)=FIRA(IGC+1,IMR)+REAL(GAR)
  80          CONTINUE
            ENDIF
  90      CONTINUE
        ENDIF
 100  CONTINUE
      IF(IPRINT.GT.2) THEN
        ALLOCATE(HDFI(NDFI),HDFP(NDFP))
        HDFI(:NDFI)=' '
        HDFP(:NDFP)=' '
        DO IFI=1,NDFI
          DO ISO=1,NBISO
            IF(FIPI(ISO,IMR).EQ.IFI) THEN
              WRITE(HDFI(IFI),'(3A4)') ISONRF(:3,ISO)
              EXIT
            ENDIF
          ENDDO
        ENDDO
        DO ISOFP=1,NDFP
          DO ISO=1,NBISO
            IF(FIFP(ISO,IMR).EQ.ISOFP) THEN
              WRITE(HDFP(ISOFP),'(3A4)') ISONRF(:3,ISO)
              EXIT
            ENDIF
          ENDDO
        ENDDO
        DO IMR=1,NMERGE
          WRITE(6,'(41H EDIHFD: FISSION YIELDS IN MERGED MIXTURE,I5,
     >    1H:/1X,12HFISSILE-----,3X,16HYIELDS----------)') IMR
          WRITE(6,'(16X,10A13)') HDFP(:NDFP)
          DO IFI=1,NDFI
            WRITE(6,'(1X,A13,1P,10E13.4/(14X,10E13.4))') HDFI(IFI),
     >      (PYIELD(IFI,ISOFP,IMR),ISOFP=1,NDFP)
          ENDDO
        ENDDO
        DEALLOCATE(HDFP,HDFI)
      ENDIF
*----
*  COMPUTE THE YIELDS
*----
      DO 130 IMR=1,NMERGE
        DO 120 IGC=1,NGCOND+1
          IF(FIRA(IGC,IMR).NE.0.0) THEN
            DO 110 ISOFP=1,NDFP
              YIELD(IGC,ISOFP,IMR)=YIELD(IGC,ISOFP,IMR)/FIRA(IGC,IMR)
 110        CONTINUE
          ENDIF
 120    CONTINUE
 130  CONTINUE
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(FIRA,SIG)
      DEALLOCATE(INDX)
      RETURN
      END
