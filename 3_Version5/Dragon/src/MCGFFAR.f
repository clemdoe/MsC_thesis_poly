*DECK MCGFFAR
      SUBROUTINE MCGFFAR(SUBSCH,K,KPN,M,N,H,NOM,NZON,XST,SP,SM,NREG,NMU,
     1           NANI,NFUNL,TRHAR,KEYFLX,KEYCUR,IMU,F,B)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Solution of transport equation on a track
* ray-tracing (anisotropic scattering,'source term isolation' on).
*
*Copyright:
* Copyright (C) 2002 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): I. Suslov and R. Le Tellier
*
*Parameters: input
* SUBSCH  track coefficients calculation subroutine.
* K       total number of volumes for which specific values
*         of the neutron flux and reactions rates are required.
* KPN     total number of unknowns in vectors F.
* M       number of material mixtures.
* N       number of elements in the current track.
* H       vector containing the lenght of the different segments of this
*         track.
* NOM     vector containing the region number of the different segments
*         of this track.
* NZON    index-number of the mixture type assigned to each volume.
* XST     macroscopic total cross section.
* SP      total source vector for + direction.
* SM      total source vector for - direction.
* NREG    number of volumes.
* NMU     order of the polar quadrature set.
* NANI    scattering anisotropy (=1 for isotropic scattering).
* NFUNL   number of moments of the flux (in 2D : NFUNL=NANI*(NANI+1)/2).
* TRHAR   spherical harmonics components for this angle in the plane.
* KEYFLX  position of flux elements in PHI vector.
* KEYCUR  position of current elements in PHI vector.
* IMU     azimuthal angle corresponding to this track.
*
*Parameters: input/output
* F       vector containing the zonal scalar flux.
*
*Parameters: scratch
* B       undefined.
*
*-----------------------------------------------------------------------
*
      IMPLICIT NONE
*----
*  SUBROUTINE ARGUMENTS
*----
      INTEGER K,KPN,M,N,NOM(N),NZON(K),NMU,NFUNL,NREG,
     1 KEYFLX(NREG,NFUNL),KEYCUR(K-NREG),IMU,NANI
      REAL XST(0:M),TRHAR(NMU,NFUNL,2)
      DOUBLE PRECISION H(N),SP(N),SM(N),F(KPN),B(N)
      EXTERNAL SUBSCH
*----
* LOCAL VARIABLES
*---
      DOUBLE PRECISION F0,FAS,RP,RM
      INTEGER I,NOMI,IND,IND1,INDN,JF,J,NOMJ
*----
*     Calculation of the coefficients for this track.
*----
*          MCGSCA: Step-Characteristics Scheme with Tabulated Exponentials
*          MCGDDF: Diamond-Differencing Scheme
*          MCGSCE: Step-Characteristics Scheme with Exact Exponentials
      IF(NANI.LE.0) CALL XABORT('MCGFFA: INVALID VALUE OF NANI.')
      CALL SUBSCH(N,K,M,NOM,NZON,H,XST,B)
*----
*     Summation along the track in both directions
*----
*     incoming flux in + direction
      IND1=KEYCUR(-NOM(1))
      RP=SP(1)
*     incoming flux in - direction
      INDN=KEYCUR(-NOM(N))
      RM=SM(N)
      DO I=2,N-1
*     + direction
         NOMI=NOM(I)
         FAS=SP(I)
         F0=B(I)*RP
         RP=RP+B(I)*(FAS-XST(NZON(NOMI))*RP)
         DO JF=1,NFUNL
            IND=KEYFLX(NOMI,JF)
            F(IND)=F(IND)+F0*TRHAR(IMU,JF,1)
         ENDDO
*     - direction
         J=N+1-I
         NOMJ=NOM(J)
         FAS=SM(J)
         F0=B(J)*RM
         RM=RM+B(J)*(FAS-XST(NZON(NOMJ))*RM)
         DO JF=1,NFUNL
            IND=KEYFLX(NOMJ,JF)
            F(IND)=F(IND)+F0*TRHAR(IMU,JF,2)
         ENDDO
      ENDDO
*     outgoing flux in + direction
      F(INDN)=F(INDN)+RP
*     outgoing flux in - direction
      F(IND1)=F(IND1)+RM
*
      RETURN
      END
