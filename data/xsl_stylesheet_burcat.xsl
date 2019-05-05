<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output indent="yes" method="xml"/>
  <xsl:strip-space elements="*"/>

  <xsl:template match="specie/phase/phase">
    <data>
      <cas_no><xsl:value-of select="../../@CAS"/></cas_no>
      <phase><xsl:value-of select="."/></phase>
      <formula_name_structure><xsl:value-of select="../../formula_name_structure/formula_name_structure_1"/></formula_name_structure>
      <reference><xsl:value-of select="../../reference/reference_1"/></reference>
      <hf298><xsl:value-of select="../../hf298/hf298_1"/></hf298>
      <max_lst_sq_error><xsl:value-of select="../../max_lst_sq_error/max_lst_sq_error_1"/></max_lst_sq_error>
      <formula><xsl:value-of select="../formula"/></formula>
      <source><xsl:value-of select="../source"/></source>
      <date><xsl:value-of select="../date"/></date>
      <range_tmin_to_1000><xsl:value-of select="../temp_limit/@low"/></range_tmin_to_1000>
      <range_1000_to_tmax><xsl:value-of select="../temp_limit/@high"/></range_1000_to_tmax>
      <molecular_weight><xsl:value-of select="../molecular_weight"/></molecular_weight>
      <hf298_div_r><xsl:value-of select="../coefficients/hf298_div_r"/></hf298_div_r>
      <a1_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a1']"/></a1_low>
      <a2_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a2']"/></a2_low>
      <a3_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a3']"/></a3_low>
      <a4_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a4']"/></a4_low>
      <a5_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a5']"/></a5_low>
      <a6_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a6']"/></a6_low>
      <a7_low><xsl:value-of select="../coefficients/range_Tmin_to_1000/coef[@name='a7']"/></a7_low>

      <a1_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a1']"/></a1_high>
      <a2_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a2']"/></a2_high>
      <a3_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a3']"/></a3_high>
      <a4_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a4']"/></a4_high>
      <a5_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a5']"/></a5_high>
      <a6_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a6']"/></a6_high>
      <a7_high><xsl:value-of select="../coefficients/range_1000_to_Tmax/coef[@name='a7']"/></a7_high>
    </data>
  </xsl:template>       

</xsl:stylesheet>