*DECK SNDSA
      SUBROUTINE SNDSA (KPSYS,INCONV,NGIND,IPTRK,IMPX,NGEFF,NREG,
     1 NBMIX,NUN,MAT,VOL,KEYFLX,KEYSPN,NUNSA,IELEMSA,ZCODE,
     2 FUNOLD,FUNKNO,NHEX)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Perform a synthetic acceleration using BIVAC (2D) or TRIVAC (3D)    
* for the discrete ordinates (SN) method using an SPn approximation.
*
*Copyright:
* Copyright (C) 2020 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): A. A. Calloo, A. Hebert and N. Martin
*
*Parameters: input
* KPSYS   pointer to the assembly matrices. KPSYS is an array of
*         directories.
* INCONV  energy group convergence flag (set to .FALSE. if converged).
* NGIND   energy group indices assign to the NGEFF set.
* IPTRK   pointer to the tracking (L_TRACK signature).
* IMPX    print flag (equal to zero for no print).
* NGEFF   number of energy groups processed in parallel.
* NREG    total number of regions for which specific values of the
*         neutron flux and reactions rates are required.
* NBMIX   number of mixtures.
* NUN     total number of unknowns in vectors FUNKNO.
* MAT     index-number of the mixture type assigned to each volume.
* VOL     volumes.
* KEYFLX  position of averaged flux elements in FUNKNO vector.
* KEYSPN  position of averaged flux elements for DSA correction.
* NUNSA   number of unknowns in BIVAC/TRIVAC.
* IELEMSA degree of the RT spatial approximation for the DSA.
* ZCODE   albedos.
* FUNOLD  SN unknown vector at iteration kappa.
* NHEX    number of hexagon.
*
*Parameters: input/output
* FUNKNO  SN unknown vector at iteration kappa+1/2 (IN) and at 
*         iteration kappa+1,i.e., with DSA correction (OUT).
*
*Comments: 
* FUNHLF is SN unknown vector at iteration kappa+1/2.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) KPSYS(NGEFF),IPTRK
      INTEGER     NGEFF,NGIND(NGEFF),IMPX,NREG,NBMIX,NUN,
     >            MAT(NREG),KEYFLX(NREG),KEYSPN(NREG),NUNSA,NHEX,
     >            IELEMSA
      LOGICAL     INCONV(NGEFF)
      REAL        VOL(NREG),ZCODE(6),FUNOLD(NUN,NGEFF),FUNKNO(NUN,NGEFF)
*----
*  LOCAL VARIABLES
*----
      PARAMETER (IUNOUT=6,NSTATE=40,PI=3.141592654)
      INTEGER IPAR(NSTATE),NLOZH,SPLTL,REM,SBMSH
      LOGICAL LSHOOT
      REAL    RAT0
*
      INTEGER, ALLOCATABLE, DIMENSION(:) :: TMPKEY,ORIKEY
      REAL, ALLOCATABLE, DIMENSION(:) :: SGAS
      REAL, ALLOCATABLE, DIMENSION(:,:) :: FUNSA,SUNSA,FUNHLF
*
      TYPE(C_PTR) DU_PTR,DE_PTR,W_PTR,DZ_PTR,U_PTR
      REAL, POINTER, DIMENSION(:) :: DU,DE,W,DZ,U
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(FUNSA(NUNSA,NGEFF),SUNSA(NUNSA,NGEFF),FUNHLF(NUN,NGEFF))
*----
*  RECOVER TRACKING INFORMATION
*----
      CALL LCMGET(IPTRK,'STATE-VECTOR',IPAR)
      ITYPE=IPAR(6)
      NSCT=IPAR(7)
      IELEM=IPAR(8)
      NDIM=IPAR(9)
      LX=IPAR(12)
      LY=IPAR(13)
      LZ=IPAR(14)
      ISOLVSA=IPAR(33)
      ISPLH=1
      IF((ITYPE.EQ.8).OR.(ITYPE.EQ.9)) ISPLH=IPAR(26)
      NLEG=0
      LL4=0
      IF(ITYPE.EQ.2) THEN
         NLEG=IELEM
         LL4 =LX*NSCT*NLEG
      ELSE IF((ITYPE.EQ.5).OR.(ITYPE.EQ.8)) THEN
         NLEG=IELEM*IELEM
         LL4 =LY*LX*NSCT*NLEG
      ELSE IF((ITYPE.EQ.7).OR.(ITYPE.EQ.9)) THEN
         NLEG=IELEM*IELEM*IELEM
         LL4 =LZ*LY*LX*NSCT*NLEG
      ELSE
         CALL XABORT('SNDSA: TYPE OF DISCRETIZATION NOT IMPLEMENTED.')
      ENDIF
*----
*  INITIALISE DSA FLUX AND SOURCE ARRAYS.
*----
      FUNSA(:NUNSA,:NGEFF)=0.0
      SUNSA(:NUNSA,:NGEFF)=0.0
*----
*  LOOP OVER ENERGY GROUPS.
*----
      DO 30 IING=1,NGEFF
      IF(.NOT.INCONV(IING)) GO TO 30
      IF(IMPX.GT.1) WRITE(IUNOUT,'(/24H SNDSA: PROCESSING GROUP,I5,
     1 6H WITH ,A,1H.)') NGIND(IING),'SN/DSA'
*----
*  RECOVER WITHIN-GROUP SCATTERING CROSS SECTION.
*----
      CALL LCMLEN(KPSYS(IING),'DRAGON-TXSC',ILONG,ITYLCM)
      IF(ILONG.NE.NBMIX+1) CALL XABORT('SNDSA: INVALID TXSC LENGTH.')
      CALL LCMLEN(KPSYS(IING),'DRAGON-S0XSC',ILONG,ITYLCM)
      NANI=ILONG/(NBMIX+1)
      ALLOCATE(SGAS(ILONG))
      CALL LCMGET(KPSYS(IING),'DRAGON-S0XSC',SGAS)
*----
*  REBUILD KEYFLX FOR HEXAGONAL CASES
*----
      ! NLOZH - num. of loz. per hexagon
      ! SBMSH - num. of submeshes per lozenge (integer)
      ! SPLTL - split of the lozenge (ISPLH)
      IF((ITYPE.EQ.8).OR.(ITYPE.EQ.9))THEN
         ALLOCATE(TMPKEY(NREG),ORIKEY(NREG))
         TMPKEY(:) = 0
         ORIKEY(1:NREG) = KEYFLX(1:NREG)
         IND = 0
         JND = 0
         NLOZH  = 3*ISPLH**2
         SBMSH  = NLOZH/3
         SPLTL  = ISPLH
         DO IZ=1,LZ
         DO IH=1,NHEX
            DO IM=1,SBMSH
               REM=MOD(IM-1,SPLTL)
               IF((REM.EQ.0).AND.(SBMSH.NE.1))THEN
                  JND = (IH-1)*NLOZH + SBMSH - (IM/SPLTL)
                  JND = JND + (IZ-1)*LX
               ELSEIF((REM.NE.0).AND.(SBMSH.NE.1))THEN
                  JND = JND - (SBMSH*3) - SPLTL
               ENDIF
               DO ILZ=1,3
                  IND = (IZ-1)*LX +(IH-1)*NLOZH +(IM-1)*3 +(ILZ-1) +1
                  IF(SBMSH.EQ.1) JND = IND
                  TMPKEY(IND) = KEYFLX(JND)
                  JND = JND + SBMSH
               ENDDO
            ENDDO
         ENDDO
         ENDDO
         KEYFLX(:) = TMPKEY(:)
         DEALLOCATE(TMPKEY)
      ENDIF
*----
*  COMPUTE THE SOURCE OF THE DSA EQUATION. 
*  Equivalency between moments of the flux for SN and SPn needs to be
*  verified.
*----
      DO 20 IR=1,NREG
      IBM=MAT(IR)
      IF(IBM.LE.0) GO TO 20
      SIGS=SGAS(IBM+1)
      DO 10 IEL=1,(IELEMSA**NDIM)
      IND=KEYFLX(IR)+IEL-1
      JND=KEYSPN(IR)+IEL-1
      SUNSA(JND,IING)=SUNSA(JND,IING)+SIGS*(FUNKNO(IND,IING)-
     >   FUNOLD(IND,IING))
  10  CONTINUE       
  20  CONTINUE      
      DEALLOCATE(SGAS)
  30  CONTINUE
*----
*  SOLVE THE SA EQUATION USING A P1 METHOD.
*----
      CALL LCMSIX(IPTRK,'DSA',1)

      IF((ITYPE.EQ.7).OR.(ITYPE.EQ.9)) THEN
         CALL TRIFLV(KPSYS,INCONV,NGIND,IPTRK,IMPX,1,NGEFF,NREG,NUNSA,
     >      MAT,VOL,KEYSPN,FUNSA,SUNSA)
      ELSE
         IF(ISOLVSA.EQ.1)THEN
            CALL PNFLV(KPSYS,INCONV,NGIND,IPTRK,IMPX,1,NGEFF,NREG,NBMIX,
     >         NUNSA,MAT,VOL,KEYSPN,FUNSA,SUNSA)
         ELSEIF(ISOLVSA.EQ.2)THEN
            CALL TRIFLV(KPSYS,INCONV,NGIND,IPTRK,IMPX,1,NGEFF,NREG,
     >         NUNSA,MAT,VOL,KEYSPN,FUNSA,SUNSA)
         ENDIF
      ENDIF
      CALL LCMSIX(IPTRK,' ',2)
*----
*  LOOP OVER ENERGY GROUPS.
*----
      RAT0=0.0
      DO 400 IING=1,NGEFF
      IF(.NOT.INCONV(IING)) GO TO 400
*--------
*  Upgrade zeroth and higher moments of the P0 SN solution,
*  ie, isotropic fluxes
      FUNHLF(:,IING)=FUNKNO(:,IING)
      DO 171 IR=1,NREG
      IF(MAT(IR).LE.0) GO TO 171
      DO IEL=1,NLEG
         RAT0=0.0
         IF(IEL.EQ.1)THEN
            INDSN=KEYFLX(IR)
            INDPN=KEYSPN(IR)
            FUNKNO(INDSN,IING)=FUNKNO(INDSN,IING)+FUNSA(INDPN,IING)
            IF(FUNHLF(INDSN,IING).LT.1.0E-7)THEN
               RAT0 =0.0
            ELSE
               RAT0 =FUNSA(INDPN,IING)/FUNHLF(INDSN,IING)
            ENDIF
         ELSE
            INDSN=KEYFLX(IR)+IEL-1 
            FUNKNO(INDSN,IING)=(1+RAT0)*FUNHLF(INDSN,IING)
         ENDIF
      ENDDO
 171  CONTINUE
*--------
* Upgrade zeroth and higher moments of the non-P0 SN solution,
* ie, anisotropic fluxes
      DO 172 IR=1,NREG
      IF(MAT(IR).LE.0) GO TO 172
      DO IEL=1,NLEG
         DO IK =1,NSCT
            IF(IK.EQ.1)THEN
               INDSN=KEYFLX(IR)
               INDPN=KEYSPN(IR)
               IF(FUNHLF(INDSN,IING).LT.1.0E-7)THEN
                  RAT0 =0.0
               ELSE
                  RAT0 =FUNSA(INDPN,IING)/FUNHLF(INDSN,IING)
               ENDIF
            ELSE
               INDSN=KEYFLX(IR)+IEL-1 + ((IK-1)*NLEG)
               FUNKNO(INDSN,IING)=(1+RAT0)*FUNHLF(INDSN,IING)
            ENDIF
         ENDDO
      ENDDO
 172  CONTINUE
*--------
* UPGRADE BOUNDARY SURFACE FLUX IN 1D CARTESIAN CASES
*--------
      LSHOOT=.TRUE.
      IF(IPAR(30).EQ.0) LSHOOT=.FALSE.
      IF((ITYPE.EQ.2).AND.(.NOT.LSHOOT)) THEN  
         CALL LCMLEN(IPTRK,'U',NLF,ITYLCM)
         CALL LCMGPD(IPTRK,'U',U_PTR)
         CALL LCMGPD(IPTRK,'W',W_PTR)
         CALL C_F_POINTER(U_PTR,U,(/ NPQ /))
         CALL C_F_POINTER(W_PTR,W,(/ NPQ /))
         IR=0
         DO II=1,LX
         IR=IR+1
         IF(MAT(IR).EQ.0) GO TO 950
         IF(VOL(IR).EQ.0.0) GO TO 950
*******XNEI-
         IF((II.EQ.1).AND.(ZCODE(1).NE.0.0)) THEN
            FHLF=0.0
            FONE=0.0
            SG=1.0
            DO IE=1,IELEM
               INDSN=KEYFLX(IR)+IE-1
               FHLF = FHLF+SG*SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SG*SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
               SG=-SG
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            DO M=1,NLF
            IF(U(M).LT.0.0) THEN
               IND1=LX*NSCT*IELEM + 1
               INDB=LX*NSCT*IELEM + M
               BHLF=BHLF+W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(1))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M) * FUNHLF(INDB,IING)/FUNHLF(IND1,IING) )
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NLF
            IF(U(M).LT.0.0) THEN
               IND1=LX*NSCT*IELEM + 1
               INDB=LX*NSCT*IELEM + M
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDIF
*******XNEI+
         IF((II.EQ.LX).AND.(ZCODE(2).NE.0.0)) THEN
            FHLF=0.0
            FONE=0.0
            DO IE=1,IELEM
               INDSN=KEYFLX(IR)+IE-1
               FHLF = FHLF+SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            DO M=1,NLF
            IF(U(M).GT.0.0) THEN
               IND1=LX*NSCT*IELEM + NLF
               INDB=LX*NSCT*IELEM + M
               BHLF=BHLF+W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(2))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M) * FUNHLF(INDB,IING)/FUNHLF(IND1,IING) )
               ! TOTW=TOTW+ABS(W(M))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NLF
            IF(U(M).GT.0.0) THEN
               IND1=LX*NSCT*IELEM + NLF
               INDB=LX*NSCT*IELEM + M
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDIF
  950    CONTINUE
         ENDDO 
*--------
* UPGRADE BOUNDARY SURFACE FLUX IN 2D CARTESIAN CASES
*--------
      ELSEIF(ITYPE.EQ.5) THEN  
         CALL LCMLEN(IPTRK,'DU',NPQ,ITYLCM)
         CALL LCMGPD(IPTRK,'DU',DU_PTR)
         CALL LCMGPD(IPTRK,'DE',DE_PTR)
         CALL LCMGPD(IPTRK,'W',W_PTR)
         CALL C_F_POINTER(DU_PTR,DU,(/ NPQ /))
         CALL C_F_POINTER(DE_PTR,DE,(/ NPQ /))
         CALL C_F_POINTER(W_PTR,W,(/ NPQ /))
         IR=0
         DO 161 JJ=1,LY
         DO 160 II=1,LX
         IR=IR+1
         IF(MAT(IR).EQ.0) GO TO 160
         IF(VOL(IR).EQ.0.0) GO TO 150
*******XNEI-
         IF((II.EQ.1).AND.(ZCODE(1).NE.0.0)) THEN
         DO JE=1,IELEM
            FHLF=0.0
            FONE=0.0
            SG=1.0
            DO IE=1,IELEM
               INDSN=KEYFLX(IR)+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SG*SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SG*SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
               SG=-SG
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0
            DO M=1,NPQ
            IF((DU(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1 = LL4 + (ICNT-1)*LY*IELEM + (JJ-1)*IELEM + JE
               INDB = LL4 +    (M-1)*LY*IELEM + (JJ-1)*IELEM + JE
               BHLF=BHLF + 2.0*W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(1))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(2.0*W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DU(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1 = LL4 + (ICNT-1)*LY*IELEM + (JJ-1)*IELEM + JE
               INDB = LL4 +    (M-1)*LY*IELEM + (JJ-1)*IELEM + JE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDIF
*******XNEI+
         IF((II.EQ.LX).AND.(ZCODE(2).NE.0.0)) THEN
         DO JE=1,IELEM
            FHLF=0.0
            FONE=0.0
            DO IE=1,IELEM
               INDSN=KEYFLX(IR)+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0
            DO M=1,NPQ
            IF((DU(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1 = LL4 + (ICNT-1)*LY*IELEM + (JJ-1)*IELEM + JE
               INDB = LL4 +    (M-1)*LY*IELEM + (JJ-1)*IELEM + JE
               BHLF=BHLF + 2.0*W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(2))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(2.0*W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DU(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1 = LL4 + (ICNT-1)*LY*IELEM + (JJ-1)*IELEM + JE
               INDB = LL4 +    (M-1)*LY*IELEM + (JJ-1)*IELEM + JE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDIF
******XNEJ-
         IF((JJ.EQ.1).AND.(ZCODE(3).NE.0.0)) THEN
         DO IE=1,IELEM
            FHLF=0.0
            FONE=0.0
            SG=1.0
            DO JE=1,IELEM
               INDSN=KEYFLX(IR)+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SG*SQRT(REAL(2*JE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SG*SQRT(REAL(2*JE-1))*FUNKNO(INDSN,IING)
               SG=-SG
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0
            DO M=1,NPQ
            IF((DE(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+IELEM*LY*NPQ+(ICNT-1)*LX*IELEM+(II-1)*IELEM+IE
               INDB=LL4+IELEM*LY*NPQ+   (M-1)*LX*IELEM+(II-1)*IELEM+IE
               BHLF=BHLF + 2.0*W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(3))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(2.0*W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DE(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+IELEM*LY*NPQ+(ICNT-1)*LX*IELEM+(II-1)*IELEM+IE
               INDB=LL4+IELEM*LY*NPQ+   (M-1)*LX*IELEM+(II-1)*IELEM+IE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDIF
*****XNEJ+
         IF((JJ.EQ.LY).AND.(ZCODE(4).NE.0.0)) THEN
         DO IE=1,IELEM
            FHLF=0.0
            FONE=0.0
            DO JE=1,IELEM
               INDSN=KEYFLX(IR)+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SQRT(REAL(2*JE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SQRT(REAL(2*JE-1))*FUNKNO(INDSN,IING)
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0
            DO M=1,NPQ
            IF((DE(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+IELEM*LY*NPQ+(ICNT-1)*LX*IELEM+(II-1)*IELEM+IE
               INDB=LL4+IELEM*LY*NPQ+   (M-1)*LX*IELEM+(II-1)*IELEM+IE
               BHLF=BHLF + 2.0*W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(4))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(2.0*W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DE(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+IELEM*LY*NPQ+(ICNT-1)*LX*IELEM+(II-1)*IELEM+IE
               INDB=LL4+IELEM*LY*NPQ+   (M-1)*LX*IELEM+(II-1)*IELEM+IE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDIF
  150    CONTINUE
  160    CONTINUE   
  161    CONTINUE   
*--------
* UPGRADE BOUNDARY SURFACE FLUX IN 3D CARTESIAN CASES
*--------
      ELSE IF(ITYPE.EQ.7) THEN   
         CALL LCMLEN(IPTRK,'DU',NPQ,ITYLCM)
         CALL LCMGPD(IPTRK,'DU',DU_PTR)
         CALL LCMGPD(IPTRK,'DE',DE_PTR)
         CALL LCMGPD(IPTRK,'DZ',DZ_PTR)
         CALL LCMGPD(IPTRK,'W',W_PTR)         
         CALL C_F_POINTER(DU_PTR,DU,(/ NPQ /))
         CALL C_F_POINTER(DE_PTR,DE,(/ NPQ /))
         CALL C_F_POINTER(W_PTR,W,(/ NPQ /))
         CALL C_F_POINTER(DZ_PTR,DZ,(/ NPQ /))
         IR=0
         DO 182 KK=1,LZ
         DO 181 JJ=1,LY
         DO 180 II=1,LX
         IR=IR+1
         IF(MAT(IR).EQ.0) GO TO 180
         IF(VOL(IR).EQ.0.0) GO TO 180
******** XNEI-
         IF((II.EQ.1).AND.(ZCODE(1).NE.0.0)) THEN 
         DO KE=1,IELEM
         DO JE=1,IELEM
            FHLF=0.0
            FONE=0.0
            SG=1.0
            DO IE=1,IELEM
               INDSN=KEYFLX(IR)+(KE-1)*IELEM**2+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SG*SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SG*SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
               SG=-SG
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0     
            DO M=1,NPQ
            IF((DU(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+ (ICNT-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               INDB=LL4+    (M-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               BHLF=BHLF + W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(1))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DU(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+ (ICNT-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               INDB=LL4+    (M-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDIF
******** XNEI+
         IF((II.EQ.LX).AND.(ZCODE(2).NE.0.0)) THEN
         DO KE=1,IELEM
         DO JE=1,IELEM
            FHLF=0.0
            FONE=0.0
            DO IE=1,IELEM
               INDSN=KEYFLX(IR)+(KE-1)*IELEM**2+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0      
            DO M=1,NPQ
            IF((DU(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+ (ICNT-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               INDB=LL4+    (M-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               BHLF=BHLF + W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(2))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DU(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+ (ICNT-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               INDB=LL4+    (M-1)*LY*LZ*IELEM**2 + (KK-1)*LY*IELEM**2 
     >            + (JJ-1)*IELEM**2 + (KE-1)*IELEM + JE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDIF
***********XNEJ-
         IF((JJ.EQ.1).AND.(ZCODE(3).NE.0.0)) THEN
         DO KE=1,IELEM
         DO IE=1,IELEM
            FHLF=0.0
            FONE=0.0
            SG=1.0
            DO JE=1,IELEM
               INDSN=KEYFLX(IR)+(KE-1)*IELEM**2+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SG*SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SG*SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
               SG=-SG
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0   
            DO M=1,NPQ
            IF((DE(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + (ICNT-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 +    (M-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               BHLF=BHLF + W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(3))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DE(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + (ICNT-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 +    (M-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDIF
*******XNEJ +
         IF((JJ.EQ.LY).AND.(ZCODE(4).NE.0.0)) THEN
         DO KE=1,IELEM
         DO IE=1,IELEM
            FHLF=0.0
            FONE=0.0
            DO JE=1,IELEM
               INDSN=KEYFLX(IR)+(KE-1)*IELEM**2+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0   
            DO M=1,NPQ
            IF((DE(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + (ICNT-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 +    (M-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               BHLF=BHLF + W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(4))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)
            
            DO M=1,NPQ
            IF((DE(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + (ICNT-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 +    (M-1)*LX*LZ*IELEM**2 
     >            + (KK-1)*LX*IELEM**2 + (II-1)*IELEM**2 + (KE-1)*IELEM 
     >            + IE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDIF
********* XNEK -
         IF((KK.EQ.1).AND.(ZCODE(5).NE.0.0)) THEN
         DO JE=1,IELEM
         DO IE=1,IELEM
            FHLF=0.0
            FONE=0.0
            SG=1.0
            DO KE=1,IELEM
               INDSN=KEYFLX(IR)+(KE-1)*IELEM**2+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SG*SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SG*SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
               SG=-SG
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0   

            DO M=1,NPQ
            IF((DZ(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            + (ICNT-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            +    (M-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               BHLF=BHLF + W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(5))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DZ(M).LT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            + (ICNT-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            +    (M-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDIF
********** XNEK +
         IF((KK.EQ.LZ).AND.(ZCODE(6).NE.0.0)) THEN
         DO JE=1,IELEM
         DO IE=1,IELEM
            FHLF=0.0
            FONE=0.0
            DO KE=1,IELEM
               INDSN=KEYFLX(IR)+(KE-1)*IELEM**2+(JE-1)*IELEM+IE-1
               FHLF = FHLF+SQRT(REAL(2*IE-1))*FUNHLF(INDSN,IING)
               FONE = FONE+SQRT(REAL(2*IE-1))*FUNKNO(INDSN,IING)
            ENDDO
            BHLF=0.0
            BONE=0.0
            TOTW=0.0
            ICNT=0   

            DO M=1,NPQ
            IF((DZ(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IF(ICNT.EQ.0) ICNT=M
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            + (ICNT-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            +    (M-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               BHLF=BHLF + W(M)*FUNHLF(INDB,IING)*1.0*(ZCODE(6))
               IF(FUNHLF(IND1,IING).NE.0.0)
     >          TOTW=TOTW+(W(M)*FUNHLF(INDB,IING)/FUNHLF(IND1,IING))
            ENDIF
            ENDDO

            IF(FHLF.NE.0.0) BONE=FONE*(BHLF/FHLF)

            DO M=1,NPQ
            IF((DZ(M).GT.0.0).AND.(W(M).NE.0.0)) THEN
               IND1=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            + (ICNT-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               INDB=LL4+ LY*LZ*NPQ*IELEM**2 + LX*LZ*NPQ*IELEM**2 
     >            +    (M-1)*LX*LY*IELEM**2 + (JJ-1)*LX*IELEM**2 
     >            + (II-1)*IELEM**2 + (JE-1)*IELEM + IE
               IF(FUNHLF(INDB,IING).EQ.0.0)THEN
                  FUNKNO(INDB,IING)=0.0
               ELSE
                  IF(FUNHLF(IND1,IING).NE.0.0)THEN
                     FUNKNO(INDB,IING)= (BONE/TOTW)*
     >                FUNHLF(INDB,IING)/FUNHLF(IND1,IING)
                  ELSE
                     FUNKNO(INDB,IING)=FUNHLF(INDB,IING)
                  ENDIF
               ENDIF
            ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDIF
  180    CONTINUE
  181    CONTINUE
  182    CONTINUE
      ENDIF
*--------
* PRINT COMPLETE UNKNOWN VECTOR.
*--------
      IF(IMPX.GT.4) THEN
         WRITE(IUNOUT,700) IING
         WRITE(IUNOUT,'(1P,4(5X,E15.7))') (FUNKNO(:,IING))
      ENDIF
*
  400 CONTINUE
*----
*  RECUPERATE ORIGINAL KEYFLX FOR HEXAGONAL CASES
*----
      IF((ITYPE.EQ.8).OR.(ITYPE.EQ.9))THEN
         KEYFLX(1:NREG) = ORIKEY(1:NREG)
         DEALLOCATE(ORIKEY)
      ENDIF
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(SUNSA,FUNSA,FUNHLF)
      RETURN
  700 FORMAT(//40H SNDSA: A F T E R    D S A    C O R R. (,I5,3H ):)
      END
